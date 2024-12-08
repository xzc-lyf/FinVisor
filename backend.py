import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from RAG import (
    SentenceTransformersEmbedding, initialize_vector_store, PromptTemplate,
    MultiAgentRAG, EmbeddingAgent, RetrievalAgent, QAAgent, CriticAgent, attain_paths_of_all_files,
    load_and_process_files
)

# Initialize FastAPI app
app = FastAPI()


# Define request and response models
class QueryRequest(BaseModel):
    queries: List[str]


class QueryResponse(BaseModel):
    query: str
    answer: str


# Initialize system components
def initialize_system():
    logging.info("Initializing system components...")

    # Load files and preprocess text
    files_path_lists = attain_paths_of_all_files(
        "/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/CUHK/CMSC5720I-Project/LLM/data"
    )
    texts = load_and_process_files(files_path_lists)

    # Initialize embedding model, vector store, and retriever
    embedding_model = SentenceTransformersEmbedding("sentence-transformers/all-MiniLM-L12-v2")
    vector_store = initialize_vector_store(texts, embedding_model, persist_directory="./vector_store")
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
            You are a helpful assistant that answers questions based on provided context.

            The following documents were retrieved to help answer the question:
            {context}

            Question: {question}

            Answer: If the answer can be found in the context, provide it. If the answer cannot be determined from the context, respond with "Cannot find the context."
        """
    )

    # Initialize QA Chain
    llm = ChatOllama(model="llama3.1:latest")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Initialize agents
    embedding_agent = EmbeddingAgent(embedding_model)
    retrieval_agent = RetrievalAgent(retriever)
    qa_agent = QAAgent(qa_chain)
    critic_agent = CriticAgent(similarity_threshold=0.5)

    # Initialize Multi-Agent RAG system
    multi_agent_rag = MultiAgentRAG(embedding_agent, retrieval_agent, qa_agent, critic_agent)

    return multi_agent_rag


# Initialize multi-agent system
multi_agent_rag = initialize_system()
executor = ThreadPoolExecutor(max_workers=10)


# API Endpoints
@app.post("/query", response_model=List[QueryResponse])
async def process_queries(request: QueryRequest):
    queries = request.queries
    responses = []

    def process_single_query(query):
        try:
            answer = multi_agent_rag.process_query(query)
            return {"query": query, "answer": answer}
        except Exception as e:
            logging.error(f"Error processing query '{query}': {str(e)}")
            return {"query": query, "answer": f"Error: {str(e)}"}

    # Process queries in parallel
    results = executor.map(process_single_query, queries)
    responses.extend(results)

    return responses


@app.get("/")
def root():
    return {"message": "Welcome to the Multi-Agent RAG API!"}
