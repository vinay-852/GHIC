from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from fastapi import UploadFile, File
import json
import database
import schemas
import ml_engine

app = FastAPI(title="Zero-Shot ML Backend")

# Dependency to get DB session
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    database.init_db()
    ml_engine.ml_instance.load_model()

# --- Helper: Background Task to Save History ---
def save_prediction_to_db(db: Session, text: str, results: list):
    """Saves the query and result to SQLite without blocking the API response"""
    db_item = database.QueryHistory(
        query_text=text,
        model_results=results, # SQLAlchemy handles JSON serialization if configured or use simple list
    )
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item.id

# --- Client Endpoints ---

@app.get("/labels", response_model=List[schemas.LabelResponse])
def get_global_labels(db: Session = Depends(get_db)):
    """Serve the user the current list of global labels"""
    return db.query(database.GlobalLabel).filter(database.GlobalLabel.is_active == True).all()

@app.post("/predict", response_model=schemas.PredictResponse)
def predict(request: schemas.PredictRequest, db: Session = Depends(get_db)):
    """
    1. Takes text input + optional custom labels
    2. Merges custom labels with global DB labels
    3. Runs ML Model
    4. Saves to DB
    5. Returns Top Results
    """
    
    # 1. Fetch global labels
    global_labels_objs = db.query(database.GlobalLabel).filter(database.GlobalLabel.is_active == True).all()
    candidate_labels = [l.label for l in global_labels_objs]
    
    # 2. Add custom client labels if provided
    if request.custom_labels:
        candidate_labels.extend(request.custom_labels)
    
    # Ensure unique labels and minimum requirement
    candidate_labels = list(set(candidate_labels))
    if not candidate_labels:
        raise HTTPException(status_code=400, detail="No labels provided (Global or Custom)")

    # 3. Run Model
    try:
        results = ml_engine.ml_instance.predict(request.text, candidate_labels)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # 4. Save to DB (Synchronously here to get the ID back for the user)
    history_id = save_prediction_to_db(db, request.text, results)

    return {
        "history_id": history_id,
        "text": request.text,
        "top_results": results
    }

@app.patch("/feedback")
def report_wrong_prediction(feedback: schemas.FeedbackRequest, db: Session = Depends(get_db)):
    """User reports a prediction was wrong and provides the correct label"""
    record = db.query(database.QueryHistory).filter(database.QueryHistory.id == feedback.history_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")
    
    record.user_reported_wrong = True
    record.correct_label_provided = feedback.correct_label
    db.commit()
    
    return {"message": "Feedback received. Thank you for improving the model."}

@app.get("/history", response_model=List[schemas.HistoryResponse])
def get_history(limit: int = 10, db: Session = Depends(get_db)):
    """Serve history of clients"""
    return db.query(database.QueryHistory).order_by(database.QueryHistory.timestamp.desc()).limit(limit).all()

# --- Admin / Training Endpoints ---

@app.put("/admin/labels/{label_id}")
def update_global_label(label_id: int, label_data: schemas.LabelUpdate, db: Session = Depends(get_db)):
    """Update an existing label's name or description"""
    # 1. Find the label
    label_item = db.query(database.GlobalLabel).filter(database.GlobalLabel.id == label_id).first()
    if not label_item:
        raise HTTPException(status_code=404, detail="Label not found")

    # 2. Check for duplicates if the name is changing
    if label_item.label != label_data.label:
        existing = db.query(database.GlobalLabel).filter(database.GlobalLabel.label == label_data.label).first()
        if existing:
            raise HTTPException(status_code=400, detail="Label name already exists")

    # 3. Update fields
    label_item.label = label_data.label
    label_item.description = label_data.description
    db.commit()
    db.refresh(label_item)
    return {"message": f"Label updated to '{label_data.label}'"}

@app.delete("/admin/labels/{label_id}")
def delete_global_label(label_id: int, db: Session = Depends(get_db)):
    """Delete a label from the database"""
    label_item = db.query(database.GlobalLabel).filter(database.GlobalLabel.id == label_id).first()
    if not label_item:
        raise HTTPException(status_code=404, detail="Label not found")
    
    db.delete(label_item)
    db.commit()
    return {"message": "Label deleted successfully"}

@app.post("/admin/labels")
def add_global_label(label_data: schemas.LabelCreate, db: Session = Depends(get_db)):
    """Admin: Add a new category to the global list"""
    existing = db.query(database.GlobalLabel).filter(database.GlobalLabel.label == label_data.label).first()
    if existing:
        raise HTTPException(status_code=400, detail="Label already exists")
    
    new_label = database.GlobalLabel(label=label_data.label, description=label_data.description)
    db.add(new_label)
    db.commit()
    return {"message": f"Label '{label_data.label}' added."}

@app.post("/admin/model/swap")
def swap_model(model_name: str):
    """Admin: Change the underlying HuggingFace model"""
    try:
        ml_engine.ml_instance.reload_model(model_name)
        return {"message": f"Model swapped to {model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/admin/fine-tune")
def trigger_fine_tuning():
    """
    Admin: Trigger a fine-tuning job.
    Note: Real fine-tuning requires a dataset and GPU training loop.
    This endpoint simulates checking the 'feedback' data and starting that process.
    """
    # Practical implementation:
    # 1. Query 'QueryHistory' where user_reported_wrong=True
    # 2. Export this data to a CSV/JSON
    # 3. Start a separate Celery worker or Subprocess to run 'Trainer' API from HuggingFace
    
    return {
        "message": "Fine-tuning job triggered. The system will gather corrected data from DB and update weights in the background.",
        # "status": "Simulated"
        "status": "In Development"
    }



@app.post("/predict/bulk", response_model=schemas.BulkResponse)
async def bulk_predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a JSON file containing a list of texts.
    Format: [{"text": "..."} , {"text": "..."}]
    """
    
    # 1. Read and Parse the File
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    # Validate format (Must be a list of dicts with 'text')
    if not isinstance(data, list) or not all("text" in item for item in data):
        raise HTTPException(status_code=400, detail="JSON must be a list of objects with a 'text' key.")

    # Limit size for safety (e.g., max 100 items for this demo)
    if len(data) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit exceeded (Max 100 items).")

    # 2. Get Labels
    global_labels_objs = db.query(database.GlobalLabel).filter(database.GlobalLabel.is_active == True).all()
    candidate_labels = [l.label for l in global_labels_objs]
    
    if not candidate_labels:
        raise HTTPException(status_code=400, detail="No global labels found in database.")

    results_list = []

    # 3. Process Loop (Practical approach)
    # Note: In a massive production app, we would use 'batching' inside the ML engine,
    # but a loop is safer and easier to debug for this setup.
    for item in data:
        text_input = item['text']
        
        # Run Prediction
        try:
            # We reuse the existing engine
            predictions = ml_engine.ml_instance.predict(text_input, candidate_labels)
            top_result = predictions[0] # Get the #1 result
            
            # Save to History (Optional: remove this line if you don't want bulk clogging DB)
            h_id = save_prediction_to_db(db, text_input, predictions)
            
            results_list.append({
                "history_id": h_id,
                "text": text_input,
                "top_label": top_result['label'],
                "confidence": top_result['score'],
                "top_results": predictions[:3]
            })
        except Exception as e:
            # If one fails, we log it but don't crash the whole batch
            results_list.append({
                "history_id": -1,
                "text": text_input, 
                "top_label": "ERROR", 
                "confidence": 0.0,
                "top_results": []
            })

    return {
        "total_processed": len(results_list),
        "results": results_list
    }

@app.post("/admin/labels/bulk")
async def bulk_add_labels(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Bulk upload labels from a JSON file.
    Format: [{"label": "Sports", "description": "..."}, {"label": "Politics"}]
    """
    try:
        content = await file.read()
        data = json.loads(content)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON file.")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be a list of objects.")

    added_count = 0
    skipped_count = 0
    errors = []

    for item in data:
        label_name = item.get("label")
        description = item.get("description")

        if not label_name:
            continue

        # Check if label already exists
        existing = db.query(database.GlobalLabel).filter(database.GlobalLabel.label == label_name).first()
        
        if existing:
            skipped_count += 1
        else:
            try:
                new_label = database.GlobalLabel(label=label_name, description=description)
                db.add(new_label)
                added_count += 1
            except Exception as e:
                errors.append(f"Error adding {label_name}: {str(e)}")

    db.commit()

    return {
        "message": "Bulk processing complete",
        "added": added_count,
        "skipped": skipped_count,
        "errors": errors
    }


@app.post("/explain", response_model=schemas.ExplainResponse)
def explain_classification(request: schemas.ExplainRequest):
    """
    Generates an explanation for why a text belongs to a specific label.
    """
    try:
        # Calls the function we added to ml_engine earlier
        explanation_text = ml_engine.ml_instance.explain_prediction(request.text, request.label,request.confidence)

        return {
            "text": request.text,
            "label": request.label,
            "explanation": explanation_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))