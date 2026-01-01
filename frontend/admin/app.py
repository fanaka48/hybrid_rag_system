import streamlit as st
import requests
import os

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="Admin Dashboard - Hybrid RAG", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
if "role" not in st.session_state:
    st.session_state.role = None

def login_signup():
    tab1, tab2 = st.tabs(["Admin Login", "Admin Registration"])
    
    with tab1:
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                response = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
                if response.status_code == 200:
                    data = response.json()
                    if data["role"] != "admin":
                        st.error("Access denied. Admin only.")
                    else:
                        st.session_state.token = data["access_token"]
                        st.session_state.role = data["role"]
                        st.rerun()
                else:
                    st.error("Invalid credentials")

    with tab2:
        st.header("Register New Admin")
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            admin_secret = st.text_input("Admin Secret Key", type="password")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                data = {
                    "username": username,
                    "password": password,
                    "role": "admin",
                    "admin_secret": admin_secret
                }
                response = requests.post(f"{BASE_URL}/signup", data=data)
                if response.status_code == 200:
                    st.success("Admin created successfully! Please login.")
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

def dashboard():
    st.title("Admin Dashboard")
    st.sidebar.button("Logout", on_click=lambda: st.session_state.clear())
    
    tabs = st.tabs(["System Dashboard", "Upload Document", "Manage Documents"])
    
    headers = {"Authorization": f"Bearer {st.session_state.token}"}

    with tabs[0]:
        st.header("System Overview")
        
        # Stats
        stats_res = requests.get(f"{BASE_URL}/admin/stats", headers=headers)
        health_res = requests.get(f"{BASE_URL}/admin/health", headers=headers)
        
        if stats_res.status_code == 200 and health_res.status_code == 200:
            stats = stats_res.json()
            health = health_res.json()
            
            summary = stats["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Users", summary["users"])
            c2.metric("Documents", summary["documents"])
            c3.metric("Conversations", summary["conversations"])
            c4.metric("Messages", summary["messages"])
            
            st.divider()
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.subheader("Document Distribution")
                st.bar_chart(stats["doc_distribution"])
                
            with col_chart2:
                st.subheader("Message Activity (7d)")
                st.line_chart(stats["activity"])
            
            st.divider()
            st.subheader("System Health")
            h1, h2, h3 = st.columns(3)
            if health['ollama'] == "Up":
                h1.success(f"Ollama: {health['ollama']}")
            else:
                h1.error(f"Ollama: {health['ollama']}")
            
            h2.info(f"FAISS: {health['faiss']}")
            h3.info(f"BM25: {health['bm25']}")
            
            st.divider()
            st.subheader("Recent Logs")
            log_res = requests.get(f"{BASE_URL}/admin/logs?n=50", headers=headers)
            if log_res.status_code == 200:
                st.code(log_res.json()["logs"], language="text")
        else:
            st.error("Could not fetch system stats")
    
    with tabs[1]:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF or Text file", type=["pdf", "txt"])
        access_level = st.selectbox("Access Level", ["user", "admin"])
        
        if st.button("Upload"):
            if uploaded_file:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                data = {"access_level": access_level}
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                
                with st.spinner("Uploading and indexing..."):
                    response = requests.post(f"{BASE_URL}/admin/upload", files=files, data=data, headers=headers)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error("Upload failed")
            else:
                st.warning("Please select a file")

    with tabs[2]:
        st.header("Document Management")
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(f"{BASE_URL}/documents", headers=headers)
        if response.status_code == 200:
            docs = response.json()
            if docs:
                import pandas as pd
                df = pd.DataFrame(docs)
                # Reorder and filter columns for display
                df = df[["id", "filename", "access_level", "uploaded_at"]]
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                st.divider()
                st.subheader("Batch Actions")
                to_delete = st.multiselect("Select Documents to Delete", options=[d["id"] for d in docs], format_func=lambda x: next(d["filename"] for d in docs if d["id"] == x))
                
                if st.button("Delete Selected", type="primary"):
                    if to_delete:
                        success_count = 0
                        for doc_id in to_delete:
                            res = requests.delete(f"{BASE_URL}/admin/documents/{doc_id}", headers=headers)
                            if res.status_code == 200:
                                success_count += 1
                        st.success(f"Successfully deleted {success_count} documents")
                        st.rerun()
                    else:
                        st.warning("Please select at least one document")
            else:
                st.info("No documents found")

if not st.session_state.token:
    login_signup()
else:
    dashboard()
