from typing import List, Dict, Any, TypedDict, Annotated, Sequence
import operator
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from app.core.config import settings
from app.services.document_processor import doc_processor
from rank_bm25 import BM25Okapi
import numpy as np
from loguru import logger

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    standalone_query: str
    context: List[Document]
    answer: str
    sources: List[Dict]

class RAGPipeline:
    def __init__(self):
        self.llm = ChatOllama(
            model=settings.CHAT_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0
        )
        self.reranker = FlashrankRerank(top_n=5)
        
    async def transform_query(self, state: AgentState):
        logger.info("Transforming query for standalone version")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a conversation and a follow-up question, rephrase the follow-up question to be a standalone question."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        chain = prompt | self.llm
        history = state["messages"][:-1]
        question = state["messages"][-1].content
        
        response = await chain.ainvoke({"history": history, "question": question})
        return {"standalone_query": response.content}

    def reciprocal_rank_fusion(self, vector_results: List[Document], bm25_results: List[Document], k=60):
        fused_scores = {}
        
        # Helper to update scores
        def update_scores(results):
            for rank, doc in enumerate(results):
                content = doc.page_content
                if content not in fused_scores:
                    fused_scores[content] = {"score": 0, "doc": doc}
                fused_scores[content]["score"] += 1 / (rank + k)
        
        update_scores(vector_results)
        update_scores(bm25_results)
        
        # Sort by score
        reranked_results = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in reranked_results]

    async def retrieve(self, state: AgentState):
        query = state["standalone_query"]
        logger.info(f"Retrieving for: {query}")
        
        vector_db, bm25_chunks = await doc_processor.get_retriever_data()
        
        # Vector search
        vector_results = []
        if vector_db:
            vector_results = vector_db.similarity_search(query, k=15)
        
        # BM25 search
        bm25_results = []
        if bm25_chunks:
            tokenized_query = query.lower().split()
            texts = [doc.page_content.lower().split() for doc in bm25_chunks]
            bm25 = BM25Okapi(texts)
            scores = bm25.get_scores(tokenized_query)
            top_n = np.argsort(scores)[-15:][::-1]
            bm25_results = [bm25_chunks[i] for i in top_n if scores[i] > 0]
        
        # Reciprocal Rank Fusion
        combined = self.reciprocal_rank_fusion(vector_results, bm25_results)
        
        return {"context": combined}

    async def rerank(self, state: AgentState):
        logger.info("Reranking results")
        query = state["standalone_query"]
        docs = state["context"]
        
        if not docs:
            return {"context": []}
            
        reranked_docs = self.reranker.compress_documents(docs, query)
        return {"context": reranked_docs}

    async def generate(self, state: AgentState):
        logger.info("Generating response")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant. Use the following context to answer the user's question. 
            If you don't know the answer, just say that you don't know. 
            Always cite your sources in the format [Source: Filename].
            
            Context:
            {context}
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        
        context_str = "\n\n".join([f"Content: {d.page_content}\nSource: {d.metadata.get('filename', 'Unknown')}" for d in state["context"]])
        
        # We don't invoke LLM here directly if we want to stream tokens from the caller 
        # but the node needs to be consistent with the graph structure.
        # So we use ainvoke here to satisfy the graph's synchronous-like nodes (even if async).
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "context": context_str,
            "history": state["messages"][:-1],
            "question": state["standalone_query"]
        })
        
        sources = [{"filename": d.metadata.get('filename'), "content": d.page_content[:200]} for d in state["context"]]
        
        return {"answer": response.content, "sources": sources}

    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)
        
        workflow.set_entry_point("transform_query")
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

rag_pipeline = RAGPipeline()
graph = rag_pipeline.build_graph()
