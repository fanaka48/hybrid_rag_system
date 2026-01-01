import streamlit as st
import requests
import json

BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="User Chat - Hybrid RAG", layout="wide")

if "token" not in st.session_state:
    st.session_state.token = None
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None

def login_signup():
    tab1, tab2 = st.tabs(["Login", "Signup"])
    
    with tab1:
        st.header("Login")
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                res = requests.post(f"{BASE_URL}/token", data={"username": u, "password": p})
                if res.status_code == 200:
                    st.session_state.token = res.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Login failed")
                    
    with tab2:
        st.header("Signup")
        with st.form("signup"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Signup"):
                res = requests.post(f"{BASE_URL}/signup", data={"username": u, "password": p})
                if res.status_code == 200:
                    st.success("Account created! Please login.")
                else:
                    st.error("Signup failed")

def chat_interface():
    st.sidebar.title("Chat History")
    headers = {"Authorization": f"Bearer {st.session_state.token}"}
    
    # List conversations
    res = requests.get(f"{BASE_URL}/conversations", headers=headers)
    if res.status_code == 200:
        convs = res.json()
        for c in convs:
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            if col1.button(f"📄 {c['title'][:20]}...", key=f"conv_{c['id']}"):
                st.session_state.current_conv_id = c["id"]
                st.rerun()
            if col2.button("🗑️", key=f"del_{c['id']}"):
                requests.delete(f"{BASE_URL}/conversations/{c['id']}", headers=headers)
                if st.session_state.current_conv_id == c["id"]:
                    st.session_state.current_conv_id = None
                st.rerun()
                
    if st.sidebar.button("New Chat", use_container_width=True):
        st.session_state.current_conv_id = None
        st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.title("Hybrid RAG Chat")
    
    # Load messages
    messages = []
    if st.session_state.current_conv_id:
        res = requests.get(f"{BASE_URL}/conversations/{st.session_state.current_conv_id}/messages", headers=headers)
        if res.status_code == 200:
            messages = res.json()

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "ai" and m.get("sources_json"):
                sources = json.loads(m["sources_json"])
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(f"- **{s['filename']}**: {s['content']}...")

    # Chat input
    if prompt := st.chat_input("Ask something..."):
        with st.chat_message("human"):
            st.markdown(prompt)
        
        with st.chat_message("ai"):
            status_container = st.status("Thinking...", expanded=True)
            placeholder = st.empty()
            full_response = ""
            sources_placeholder = st.empty()
            
            data = {"message": prompt}
            if st.session_state.current_conv_id:
                data["conversation_id"] = st.session_state.current_conv_id
            
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            
            try:
                with requests.post(f"{BASE_URL}/chat", data=data, headers=headers, stream=True) as r:
                    if r.status_code == 200:
                        for line in r.iter_lines():
                            if line:
                                chunk = json.loads(line.decode())
                                
                                if "status" in chunk:
                                    status_container.update(label=chunk["status"], state="running")
                                
                                if "token" in chunk:
                                    # Once we start getting tokens, we can finish the status
                                    status_container.update(label="Response generated", state="complete", expanded=False)
                                    full_response += chunk["token"]
                                    placeholder.markdown(full_response + "▌")
                                
                                if "conversation_id" in chunk:
                                    st.session_state.current_conv_id = chunk["conversation_id"]
                                
                                if "sources" in chunk:
                                    sources = chunk["sources"]
                                    if sources:
                                        with sources_placeholder.expander("Sources"):
                                            for s in sources:
                                                st.write(f"- **{s['filename']}**: {s['content']}...")
                        
                        placeholder.markdown(full_response)
                        st.rerun()
                    else:
                        status_container.update(label="Error occurred", state="error")
                        st.error(f"Error sending message: {r.text}")
            except Exception as e:
                status_container.update(label="Connection error", state="error")
                st.error(f"Error: {e}")

if not st.session_state.token:
    login_signup()
else:
    chat_interface()
