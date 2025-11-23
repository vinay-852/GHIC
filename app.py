import streamlit as st
import requests
import pandas as pd
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"
st.set_page_config(page_title="Zero-Shot ML Dashboard", layout="wide")

# --- HELPER FUNCTIONS ---
def get_global_labels():
    try:
        response = requests.get(f"{API_URL}/labels")
        if response.status_code == 200:
            return [item['label'] for item in response.json()]
        return []
    except:
        return []

# --- SIDEBAR ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["User Client", "Admin Dashboard"])

# ==========================================
# MODE 1: USER CLIENT
# ==========================================
if app_mode == "User Client":
    st.title("🤖 Zero-Shot Classifier Client")

    # Create Tabs for Input Method
    tab_single, tab_bulk = st.tabs(["Single Input", "Bulk Upload (JSON)"])

    # ==========================
    # TAB 1: SINGLE INPUT
    # ==========================
    with tab_single:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            query_text = st.text_area("Enter text to classify:", height=150)
        
        with col2:
            st.subheader("Categories")
            global_labels = get_global_labels()
            st.info(f"Global Labels: {', '.join(global_labels)}")
            custom_labels_input = st.text_input("Add temporary labels (comma separated):")
            
        if st.button("Run Classification", type="primary"):
            if not query_text:
                st.warning("Please enter some text.")
            else:
                # Prepare payload
                custom_list = [x.strip() for x in custom_labels_input.split(",")] if custom_labels_input else []
                payload = {
                    "text": query_text,
                    "custom_labels": custom_list if custom_list else []
                }

                with st.spinner("Running Model..."):
                    try:
                        res = requests.post(f"{API_URL}/predict", json=payload)
                        if res.status_code == 200:
                            data = res.json()
                            st.success("Done!")
                            st.session_state['feedback_id_input'] = data['history_id']
                            st.markdown(f"**ID:** `{data['history_id']}`")

                            results = data['top_results']
                            top_label = results[0]['label']  # Get the #1 prediction
                            confidence = results[0]['score']
                            # --- NEW EXPLAINABILITY SECTION ---

                            with st.expander(f"🤔 Why '{top_label}'? (Explainability)"):
                                with st.spinner("Generating explanation..."):
                                    try:
                                        # 1. Construct the payload matching schemas.ExplainRequest
                                        explain_payload = {
                                            "text": query_text,
                                            "label": top_label,
                                            "confidence": confidence

                                        }
                                        st.info(f"Explain: {explain_payload}")
                                        explain_response = requests.post(f"{API_URL}/explain", json=explain_payload)

                                        # 3. Handle the response
                                        if explain_response.status_code == 200:
                                            result_data = explain_response.json()
                                            explanation = result_data.get("explanation")

                                            # Display the result
                                            st.markdown(f"**Analysis:**")
                                            st.write(explanation)
                                        else:
                                            st.error(
                                                f"Failed to get explanation. Status: {explain_response.status_code}")

                                    except Exception as e:
                                        st.error(f"An error occurred while connecting to the server: {e}")

                            results = data['top_results']
                            df = pd.DataFrame(results).set_index('label')
                            st.bar_chart(df['score'])
                            st.table(df)
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

    # ==========================
    # TAB 2: BULK UPLOAD
    # ==========================
    with tab_bulk:
        st.write("### Upload a JSON file")
        st.caption('Format: `[{"text": "example 1"}, {"text": "example 2"}]`')
        
        uploaded_file = st.file_uploader("Choose a JSON file", type="json")
        
        if "bulk_results" not in st.session_state:
            st.session_state.bulk_results = None

        if uploaded_file is not None:
            if st.button("Process Bulk File"):
                with st.spinner("Processing..."):
                    try:
                        files = {"file": ("filename.json", uploaded_file, "application/json")}
                        res = requests.post(f"{API_URL}/predict/bulk", files=files)
                        
                        if res.status_code == 200:
                            st.session_state.bulk_results = res.json()['results']
                            st.success(f"Processed {len(st.session_state.bulk_results)} items!")
                        else:
                            st.error(f"Error: {res.text}")
                    except Exception as e:
                        st.error(f"Connection Error: {e}")

         # --- DUAL VIEW RENDERER ---
        if st.session_state.bulk_results:
            st.divider()
            
            # Sub-Tabs for Viewing
            view_tab, action_tab = st.tabs(["📊 Data Table View", "✅ Interactive Feedback View"])
            
            # --- VIEW A: DATAFRAME ---
            with view_tab:
                simple_data = []
                for row in st.session_state.bulk_results:
                    simple_data.append({
                        "ID": row['history_id'],
                        "Text": row['text'],
                        "Prediction": row['top_label'],
                        "Confidence": f"{row['confidence']:.4f}"
                    })
                st.dataframe(pd.DataFrame(simple_data), use_container_width=True)

            # --- VIEW B: INTERACTIVE LIST ---
            with action_tab:
                all_labels = get_global_labels()
                
                # Headers
                h1, h2, h3, h4 = st.columns([3, 1, 1, 2])
                h1.markdown("**Text**")
                h2.markdown("**Prediction**")
                h3.markdown("**Conf.**")
                h4.markdown("**Action**")
                st.divider()

                for row in st.session_state.bulk_results:
                    row_id = row['history_id']
                    c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
                    
                    with c1: st.write(row['text'])
                    with c2: 
                        color = "green" if row['confidence'] > 0.8 else "orange"
                        st.markdown(f":{color}[{row['top_label']}]")
                    with c3: st.write(f"{row['confidence']:.2f}")
                    with c4: 
                        is_wrong = st.checkbox("Report", key=f"chk_{row_id}")

                    if is_wrong:
                        with st.container():
                            f1, f2 = st.columns([3, 1])
                            with f1:
                                correct_label = st.selectbox("Correct Label:", all_labels, key=f"sel_{row_id}")
                            with f2:
                                st.write("")
                                st.write("")
                                if st.button("Submit", key=f"btn_{row_id}"):
                                    try:
                                        requests.patch(f"{API_URL}/feedback", json={"history_id": row_id, "correct_label": correct_label})
                                        st.success("Saved!")
                                    except: st.error("Error")
                    st.markdown("---")

            # --- CSV DOWNLOAD LOGIC (TOP 3) ---
            st.divider()
            
            # Helper to flatten the JSON structure for CSV
            csv_data = []
            for item in st.session_state.bulk_results:
                row = {
                    "history_id": item['history_id'],
                    "text": item['text'],
                }
                # Flatten top 3 results
                top_res = item.get('top_results', [])
                
                # Loop 1 to 3
                for i in range(3):
                    if i < len(top_res):
                        row[f"label_{i+1}"] = top_res[i]['label']
                        row[f"score_{i+1}"] = top_res[i]['score']
                    else:
                        row[f"label_{i+1}"] = ""
                        row[f"score_{i+1}"] = 0.0
                
                csv_data.append(row)

            df_export = pd.DataFrame(csv_data)
            csv = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                "📥 Download Detailed Results (Top 3 Labels)",
                csv,
                "bulk_results_detailed.csv",
                "text/csv"
            )

    st.divider()

    # 3. FEEDBACK & HISTORY SECTION
    st.subheader("📜 History & Feedback")
    
    if st.button("Refresh History"):
        try:
            hist_res = requests.get(f"{API_URL}/history?limit=20")
            if hist_res.status_code == 200:
                st.session_state['history_data'] = hist_res.json()
            else:
                st.error("Failed to fetch history")
        except:
            st.error("Backend offline")

    if 'history_data' in st.session_state:
        # Create a cleaner table for display
        display_data = []
        for item in st.session_state['history_data']:
            # Extract top label for summary
            top_prediction = item['model_results'][0]['label'] if item['model_results'] else "N/A"
            display_data.append({
                "ID": item['id'],
                "Time": item['timestamp'],
                "Text": item['query_text'][:50] + "...",
                "Model Prediction": top_prediction,
                "Validation Result": "❌" if item['user_reported_wrong'] else "✅"
            })
        
        st.dataframe(pd.DataFrame(display_data))

        # Feedback Form
        st.write("### Report an Issue")
        all_labels = get_global_labels()
        with st.form("feedback_form"):
            c_id, c_label = st.columns([1, 3])
            with c_id:
                f_id = st.number_input("Enter History ID to correct:", min_value=1, step=1, key="feedback_id_input")
            with c_label:
                if all_labels:
                    f_correct = st.selectbox("Correct Label", options=all_labels)
                else:
                    f_correct = st.text_input("Correct Label (System offline/No Labels, type manually)")
            submitted = st.form_submit_button("Submit Correction")
            
            if submitted:
                payload = {"history_id": f_id, "correct_label": f_correct}
                res = requests.patch(f"{API_URL}/feedback", json=payload)
                if res.status_code == 200:
                    st.success("Feedback received! Thank you.")
                else:
                    st.error(f"Error: {res.text}")

# ==========================================
# MODE 2: ADMIN DASHBOARD
# ==========================================
elif app_mode == "Admin Dashboard":
    st.title("⚙️ Admin Dashboard")
    
    tab1, tab2 = st.tabs(["Manage Labels", "Model Operations"])

    # --- TAB 1: MANAGE LABELS ---
    with tab1:
        # --- SECTION A: CREATE NEW ---
        with st.expander("➕ Add New Category", expanded=False):
            st.subheader("Add New Global Category")
            with st.container():
                new_label = st.text_input("New Label Name")
                raw_desc = st.text_area("Rough Description")
                
                if "final_desc" not in st.session_state:
                    st.session_state.final_desc = ""

                col_a, col_b = st.columns(2)
                

                with col_b:
                    st.write("") 
                    st.write("") 
                    if st.button("💾 Save New Label", type="primary"):
                        desc_to_save = st.session_state.final_desc if st.session_state.final_desc else raw_desc
                        if not new_label:
                            st.error("Label name is required")
                        else:
                            payload = {"label": new_label, "description": desc_to_save}
                            res = requests.post(f"{API_URL}/admin/labels", json=payload)
                            if res.status_code == 200:
                                st.success(f"Label '{new_label}' saved!")
                                if "suggested_result" in st.session_state:
                                    del st.session_state.suggested_result
                                st.rerun() # Refresh page to show in table
                            else:
                                st.error(res.text)
        with st.expander("📤 Bulk Upload Labels", expanded=False):
            st.write("### Upload Labels via JSON")
            st.caption('Format: `[{"label": "Sports", "description": "Optional desc"}, ...]`')
            
            bulk_label_file = st.file_uploader("Choose JSON File", type="json", key="bulk_label_upload")
            
            if bulk_label_file is not None:
                if st.button("Process Bulk Labels"):
                    with st.spinner("Uploading labels..."):
                        try:
                            files = {"file": ("labels.json", bulk_label_file, "application/json")}
                            res = requests.post(f"{API_URL}/admin/labels/bulk", files=files)
                            
                            if res.status_code == 200:
                                data = res.json()
                                st.success(f"✅ Added: {data['added']}")
                                if data['skipped'] > 0:
                                    st.warning(f"⚠️ Skipped (Duplicates): {data['skipped']}")
                                if data['errors']:
                                    st.error(f"Errors: {data['errors']}")
                                
                                # Trigger refresh of the table below
                                time.sleep(1) 
                                st.rerun()
                            else:
                                st.error(f"Error: {res.text}")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")

        st.divider()

        # --- SECTION B: VIEW / EDIT / DELETE ---
        st.subheader("Manage Existing Labels")
        
        # 1. Get current labels
        try:
            labels_data = requests.get(f"{API_URL}/labels").json()
        except:
            labels_data = []
            st.error("Could not fetch labels.")

        if labels_data:
            # Convert to DataFrame for display
            df_labels = pd.DataFrame(labels_data)
            st.dataframe(df_labels, use_container_width=True)

            st.write("### Edit or Delete")
            
            label_options = {item['label']: item for item in labels_data}
            selected_label_name = st.selectbox("Select Label to Edit/Delete:", list(label_options.keys()))
            
            if selected_label_name:
                selected_item = label_options[selected_label_name]
                selected_id = selected_item["id"]

                with st.form("edit_delete_form"):
                    col_edit, col_del = st.columns([3, 1])
                    
                    with col_edit:
                        edit_name = st.text_input("Edit Name", value=selected_item['label'])
                        edit_desc = st.text_area("Edit Description", value=selected_item.get('description', ''))
                        
                        update_submitted = st.form_submit_button("Update Label")
                    
                    with col_del:
                        st.write("") # Spacer
                        st.write("")
                        pass

                    if update_submitted:
                        payload = {"label": edit_name, "description": edit_desc}
                        res = requests.put(f"{API_URL}/admin/labels/{selected_id}", json=payload)
                        if res.status_code == 200:
                            st.success("Updated successfully!")
                            st.rerun()
                        else:
                            st.error(f"Update failed: {res.text}")

                # Delete Button (Outside form to act independently)
                st.write("")
                col_warn, col_btn = st.columns([3, 1])
                with col_warn:
                    st.warning(f"⚠️ Deleting '{selected_label_name}' cannot be undone.")
                with col_btn:
                    if st.button("🗑️ Delete Label", type="secondary"):
                        res = requests.delete(f"{API_URL}/admin/labels/{selected_id}")
                        if res.status_code == 200:
                            st.success("Deleted successfully!")
                            st.rerun()
                        else:
                            st.error(f"Delete failed: {res.text}")
        else:
            st.info("No labels found in database.")
