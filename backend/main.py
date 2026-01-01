from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select
from typing import List, Optional
import os
import shutil
from datetime import datetime, timedelta

from app.core.config import settings
from app.db.session import init_db, get_session, engine
from app.models.models import User, Document, Conversation, Message
from app.core.auth import verify_password, get_password_hash, create_access_token
from app.services.document_processor import doc_processor
from app.services.rag_pipeline import graph
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger
import sys

# Setup logging
os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
logger.add(settings.LOG_FILE, rotation="10 MB", level="INFO")

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.on_event("startup")
async def on_startup():
    await init_db()
    from sqlalchemy.orm import sessionmaker
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        # Create default admin if not exists
        result = await session.execute(select(User).where(User.username == "admin"))
        admin = result.scalars().first()
        if not admin:
            admin = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                role="admin"
            )
            session.add(admin)
            await session.commit()
            logger.info("Default admin created")

@app.get("/ping")
async def ping():
    return {"message": "pong"}

# --- Auth Endpoints ---

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), session: AsyncSession = Depends(get_session)):
    logger.info(f"Login attempt for user: {form_data.username}")
    result = await session.execute(select(User).where(User.username == form_data.username))
    user = result.scalars().first()
    if not user:
        logger.warning(f"User not found: {form_data.username}")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Invalid password for user: {form_data.username}")
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    logger.info(f"Login successful for user: {form_data.username}")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)) -> User:
    from jose import jwt, JWTError
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    result = await session.execute(select(User).where(User.username == username))
    user = result.scalars().first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.post("/signup")
async def signup(
    username: str = Form(...), 
    password: str = Form(...), 
    role: str = Form("user"),
    admin_secret: Optional[str] = Form(None),
    session: AsyncSession = Depends(get_session)
):
    result = await session.execute(select(User).where(User.username == username))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    if role == "admin":
        if admin_secret != settings.ADMIN_REGISTRATION_SECRET:
            raise HTTPException(status_code=403, detail="Invalid admin secret")
    
    user = User(username=username, hashed_password=get_password_hash(password), role=role)
    session.add(user)
    await session.commit()
    return {"message": f"{role.capitalize()} created"}

# --- Document Endpoints ---

@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...),
    access_level: str = Form("user"),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload documents")
    
    file_path = os.path.join(settings.STORAGE_PATH, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    doc = Document(filename=file.filename, storage_path=file_path, access_level=access_level)
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    
    # Process and index
    chunks = await doc_processor.process_file(file_path, {"filename": file.filename, "doc_id": doc.id})
    await doc_processor.add_to_index(chunks)
    
    return {"message": f"File {file.filename} uploaded and indexed"}

@app.get("/documents", response_model=List[Document])
async def list_documents(current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    # Admins see all docs, users see only 'user' level docs
    if current_user.role == "admin":
        result = await session.execute(select(Document))
        return result.scalars().all()
    result = await session.execute(select(Document).where(Document.access_level == "user"))
    return result.scalars().all()

@app.get("/admin/stats")
async def get_stats(current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    users = (await session.execute(select(User))).scalars().all()
    docs = (await session.execute(select(Document))).scalars().all()
    msgs = (await session.execute(select(Message))).scalars().all()
    convs = (await session.execute(select(Conversation))).scalars().all()
    
    # Document distribution
    doc_levels = {"user": 0, "admin": 0}
    for d in docs:
        doc_levels[d.access_level] = doc_levels.get(d.access_level, 0) + 1
    
    # Simple message activity (last 7 days - mock for now since Message doesn't have robust date filtering in current query)
    # In a real app, you'd do: select(func.date(Message.created_at), func.count())...
    activity = {
        (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d"): len(msgs) // (i + 1)
        for i in range(7)
    }
    
    return {
        "summary": {
            "users": len(users),
            "documents": len(docs),
            "messages": len(msgs),
            "conversations": len(convs)
        },
        "doc_distribution": doc_levels,
        "activity": activity
    }

@app.get("/admin/logs")
async def get_logs(n: int = 100, current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if not os.path.exists(settings.LOG_FILE):
        return {"logs": "Log file not found"}
        
    with open(settings.LOG_FILE, "r") as f:
        lines = f.readlines()
        return {"logs": "".join(lines[-n:])}

@app.get("/admin/health")
async def health_check(current_user: User = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    import httpx
    ollama_status = "Down"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(settings.OLLAMA_BASE_URL)
            if res.status_code == 200:
                ollama_status = "Up"
    except:
        pass
        
    faiss_status = "Ready" if os.path.exists(settings.FAISS_INDEX_PATH) else "Not Found"
    bm25_status = "Ready" if os.path.exists(os.path.join(settings.STORAGE_PATH, "bm25.pkl")) else "Not Found"
    
    return {
        "ollama": ollama_status,
        "faiss": faiss_status,
        "bm25": bm25_status
    }

@app.delete("/admin/documents/{doc_id}")
async def delete_document(
    doc_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can delete documents")
    
    doc = await session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Remove from indices
    await doc_processor.delete_document(doc_id)
    
    # Remove file
    if os.path.exists(doc.storage_path):
        os.remove(doc.storage_path)
    
    # Remove from DB
    await session.delete(doc)
    await session.commit()
    
    return {"message": f"Document {doc_id} deleted"}

# --- Chat Endpoints ---

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/chat")
async def chat(
    conversation_id: Optional[int] = Form(None),
    message: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    if not conversation_id:
        conv = Conversation(user_id=current_user.id, title=message[:50])
        session.add(conv)
        await session.commit()
        await session.refresh(conv)
        conversation_id = conv.id
    else:
        conv = await session.get(Conversation, conversation_id)
        if not conv or conv.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # Save human message
    human_msg = Message(conversation_id=conversation_id, role="human", content=message)
    session.add(human_msg)
    await session.commit()

    # Get history
    result = await session.execute(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at))
    history_objs = result.scalars().all()
    history = []
    for m in history_objs:
        if m.role == "human":
            history.append(HumanMessage(content=m.content))
        else:
            history.append(AIMessage(content=m.content))

    async def event_generator():
        full_answer = ""
        sources = []
        
        # We use astream_events to get token-level updates
        # The 'generate' node is where the final answer comes from.
        # However, to keep it simple and robust with citations, 
        # we'll stream from the LLM inside the generate node or similar.
        # For now, let's yield the full result in chunks to simulate streaming if needed,
        # OR use astream_events properly.
        
        try:
            # We first yield the conversation_id so the frontend knows it
            yield json.dumps({"conversation_id": conversation_id}) + "\n"
            
            # Start the graph
            async for event in graph.astream_events({"messages": history}, version="v2"):
                kind = event["event"]
                
                # Track transition between nodes
                if kind == "on_chain_start" and event["name"] == "transform_query":
                    yield json.dumps({"status": "Transforming query..."}) + "\n"
                elif kind == "on_chain_start" and event["name"] == "retrieve":
                    yield json.dumps({"status": "Retrieving context..."}) + "\n"
                elif kind == "on_chain_start" and event["name"] == "rerank":
                    yield json.dumps({"status": "Reranking results..."}) + "\n"
                elif kind == "on_chain_start" and event["name"] == "generate":
                    yield json.dumps({"status": "Generating answer..."}) + "\n"

                # We want tokens from the 'generate' node's LLM call
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        full_answer += content
                        yield json.dumps({"token": content}) + "\n"
                
                # We want the sources from the final state
                elif kind == "on_chain_end" and event["name"] == "LangGraph":
                    # This is the final state
                    output = event["data"]["output"]
                    if "sources" in output:
                        sources = output["sources"]
            
            # Save AI message to DB
            ai_msg = Message(
                conversation_id=conversation_id, 
                role="ai", 
                content=full_answer
            )
            ai_msg.sources = sources
            session.add(ai_msg)
            await session.commit()
            
            # Yield final sources
            yield json.dumps({"sources": sources}) + "\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

@app.get("/conversations", response_model=List[Conversation])
async def list_conversations(current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Conversation).where(Conversation.user_id == current_user.id).order_by(Conversation.created_at.desc()))
    return result.scalars().all()

@app.get("/conversations/{conversation_id}/messages", response_model=List[Message])
async def get_messages(conversation_id: int, current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    result = await session.execute(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.created_at))
    return result.scalars().all()

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: int, current_user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    conv = await session.get(Conversation, conversation_id)
    if not conv or conv.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.delete(conv)
    await session.commit()
    return {"message": "Conversation deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
