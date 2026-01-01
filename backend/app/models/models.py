from datetime import datetime
from typing import List, Optional, Dict
from sqlmodel import SQLModel, Field, Relationship
import json

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str
    role: str = "user"  # admin or user
    is_active: bool = True
    
    conversations: List["Conversation"] = Relationship(back_populates="user")
    documents: List["Document"] = Relationship(back_populates="uploader")

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    storage_path: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    uploaded_by: int = Field(foreign_key="user.id")
    access_level: str = "user"  # user (everyone) or admin (only admin)
    
    uploader: User = Relationship(back_populates="documents")

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    title: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    user: User = Relationship(back_populates="conversations")
    messages: List["Message"] = Relationship(back_populates="conversation", cascade_delete=True)

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id")
    role: str  # human or ai
    content: str
    sources_json: str = Field(default="[]") # Store sources as JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    conversation: Conversation = Relationship(back_populates="messages")

    @property
    def sources(self) -> List[Dict]:
        return json.loads(self.sources_json)

    @sources.setter
    def sources(self, value: List[Dict]):
        self.sources_json = json.dumps(value)
