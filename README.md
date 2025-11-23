# 🧩 **Decode – Automated AI-Based Financial Transaction Categorisation**

### Team: **Decode**

**Members:**

1. Vinay Pepakayala
    
2. Navaneeth Kola
    
3. Teja Gorrepotu
    

---

## 🚀 **Overview**

Modern financial applications—from personal budgeting platforms to enterprise accounting systems—depend on **accurate categorisation of raw financial transaction strings** such as:

- “Starbucks 0423”
    
- “UBER TRIP”
    
- “Amazon Marketplace”
    
- “Shell Fuel Pump”
    

These raw descriptions must be mapped to categories such as **Dining, Transportation, Shopping, Fuel, Utilities**, etc.

Today, most companies rely heavily on **paid third-party categorisation APIs**, which introduce:

- Recurring costs
    
- Limited customisation
    
- Latency
    
- Vendor lock-in
    
- Privacy concerns
    

**Decode** aims to eliminate these limitations by building a fully **in-house, AI-powered, customisable, explainable, scalable** transaction categorisation system—without depending on any external API.

---

# 🎯 **Problem Statement**

Financial systems require scalable and accurate transaction classification to support budgeting, analytics, fraud monitoring, and financial planning. Outsourcing categorisation to external APIs introduces problems such as:

- High recurring costs
    
- Latency due to external calls
    
- Lack of adaptability to custom taxonomies
    
- Limited transparency
    
- Data privacy and compliance challenges
    

This project solves these challenges by developing a **standalone, high-performance ML system** that:

- Classifies transactions autonomously
    
- Achieves business-grade accuracy
    
- Explains its decisions
    
- Allows real-time taxonomy updates
    
- Supports human-in-the-loop corrections
    
- Requires **zero third-party API usage**
    

---

# 🧠 **Core Features**

### 🔹 **1. Embedding-Based Classification (Zero-Shot + Dynamic Labels)**

Instead of using fixed classifiers, the system embeds:

- The **transaction text**
    
- All **category labels from the admin panel**
    

...and performs **cosine similarity** to determine the closest category.

Benefits:

- Add/delete categories **without retraining**
    
- Flexible taxonomy for enterprises
    
- Supports 100+ categories with constant inference cost
    

---

### 🔹 **2. Explainable AI (XAI)**

Every prediction includes a **human-readable explanation** powered by a small LLM (Qwen2.5-0.5B):

> “This transaction aligns with the _Fuel_ category due to merchant semantics and transportation-related keywords.”

---

### 🔹 **3. Feedback Loop (Human-in-the-Loop Learning)**

Users can mark predictions as incorrect and provide the correct label.  
Stored as training data for **future fine-tuning**.

---

### 🔹 **4. Bulk Inference Engine**

Upload JSON with hundreds of transactions.  
Outputs:

- Predictions
    
- Confidence scores
    
- Top-3 categories
    
- Downloadable CSV
    

---

### 🔹 **5. Admin Dashboard**

Admins can:

- Add labels
    
- Edit labels
    
- Delete labels
    
- Bulk upload taxonomy
    
- Trigger simulated fine-tuning
    
- Swap embedding models
    

---

### 🔹 **6. No External API Usage**

All models run locally:

- **all-mpnet-base-v2** (embedding)
    
- **Qwen2.5-0.5B Instruct** (text explanation)
    

Ensuring:

- Data privacy
    
- Zero recurring cost
    
- Offline capability
    

---

# 🏗️ **System Architecture**

```
                ┌────────────────────────────────────────┐
                │              Streamlit UI               │
                │  - User Client (Single/Bulk)            │
                │  - Admin Dashboard                      │
                └──────────────────────┬──────────────────┘
                                       │ HTTP (REST)
                                       ▼
                ┌────────────────────────────────────────┐
                │              FastAPI Backend            │
                │  /predict       – ML inference          │
                │  /predict/bulk  – Batch processing      │
                │  /explain       – XAI generation        │
                │  /admin/labels  – Taxonomy mgmt         │
                │  /feedback      – Human correction      │
                |  more..
                └───────────────┬───────────────┬────────┘
                                │               │
                                ▼               ▼
                 ┌─────────────────────┐   ┌──────────────────────┐
                 │     ML Engine       │   │     SQLite DB         │
                 │  - MPNet Embedder   │   │  - Labels             │
                 │  - Cosine Similarity│   │  - History            │
                 │  - Qwen LLM XAI     │   │  - Feedback           │
                 └─────────────────────┘   └──────────────────────┘
```

---

# 🧬 **Technology Stack**

|Component|Technology|
|---|---|
|**Language**|Python|
|**Embedding Model**|`sentence-transformers/all-mpnet-base-v2`|
|**Explainability LLM**|Qwen/Qwen2.5-0.5B-Instruct|
|**Frameworks**|FastAPI, Pydantic, Streamlit|
|**ML Libraries**|PyTorch, Transformers|
|**Database**|SQLite (SQLAlchemy ORM)|
|**Storage**|JSON / Local dataset|
|**Deployment**|Local execution (extendable to Docker/Cloud)|

---

# 🔬 **AI / ML Methodology**

### **1. Embedding-Based Zero-Shot Classification**

We embed:

```
Transaction Text → Vector
Category Label → Vector
Cosine Similarity( text_vec , label_vec )
```

This allows:

- Custom categories
    
- Unlimited taxonomy growth
    
- No retraining required
    

---

### **2. Synthetic Dataset Fine-Tuning**

A curated synthetic dataset was created to mimic:

- Misspellings
    
- Ambiguous short transactions
    
- Merchant name variations
    
- Category edge cases
    

> Improves robustness to noisy real-world bank statements.

---

### **3. Confidence Scoring & Thresholding**

Low confidence predictions are:

- Highlighted in UI
    
- Pushed for manual review
    
- Used for future fine-tuning
    

---

### **4. Explainability via LLM**

Each prediction generates a short natural-language explanation.

---

# 📊 **Evaluation**

The system was tested on synthetic + sourced public transaction datasets.

**Metrics considered:**

- Macro F1-score
    
- Confusion matrix
    
- Confidence distribution
    
- Error clustering (semantic misclassifications)
    

The architecture consistently showed:

- **High robustness to noisy text**
    
- **Strong clustering of semantically similar merchants**
    
- **Clear separation between distant categories**
    

---

# 🛡️ **Security & Responsible AI**

### ✔ No external API calls → ensures privacy

### ✔ Local/offline inference supported

### ✔ Bias Mitigation:

- No sensitive attributes used
    
- Treats all merchants equivalently
    
- Human-in-loop correction reduces systemic drift
    

---

# ⚙️ **How to Run the Project**

## **1️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

## **2️⃣ Start FastAPI Backend**

```bash
uvicorn main:app --reload
```

Backend runs at:  
`http://127.0.0.1:8000`

## **3️⃣ Start Streamlit Frontend**

```bash
streamlit run app.py
```

UI opens at:  
`http://localhost:8501`


---

# 📂 **Project Structure**

```
├── app.py                 # Streamlit UI
├── main.py                # FastAPI backend
├── ml_engine.py           # Embedding engine + LLM XAI
├── database.py            # SQLite models + ORM
├── schemas.py             # API schemas
├── app_data.db            # Local DB
├── README.md              # Documentation
└── requirements.txt
```

---

# 📦 **Extendability Roadmap**

✔ Add FAISS vector index for large-scale search  
✔ Add real fine-tuning pipeline with HuggingFace Trainer  
✔ On-device model compression  
✔ Add mobile-ready lightweight classifier  
✔ Multi-lingual support  
✔ Dockerization

---

# 📹 **Demo & Repository**

This section is for the submission:

**GitHub Repository:** _Add your link here_  
**Demo Video:** _https://drive.google.com/drive/folders/14xRfA45jrdaJMcK7Qr4pw3Iifvwwlii6?usp=sharing_

---

# 🏁 **Conclusion**

Decode provides a **secure, cost-effective, scalable, and fully customizable AI system** for financial transaction categorisation, delivering:

- High accuracy
    
- Zero API dependency
    
- Real-time explainability
    
- Fine-grained admin control
    
- Enterprise scalability
    

A future-ready alternative to expensive third-party solutions.

---
