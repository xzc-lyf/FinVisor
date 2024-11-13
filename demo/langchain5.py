import os
import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set environment variables
os.environ['USER_AGENT'] = 'FinTechHelper/1.0 (xzco.work@gmail.com)'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# Set host and port
host = "127.0.0.1"
port = "11434"

# Initialize model
# model = OllamaLLM(base_url=f"http://{host}:{port}", model="llama3.1:latest", temperature=0)
model = ChatOllama(model="llama3.1:latest")

# Load web data
loader = WebBaseLoader(
    web_paths=['https://www.eastmoney.com/default.html'],
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=('post-header', 'post-title', 'post-content'))
    )
)
docs = loader.load()

# Example text to split
text = "融资余额创9年新高。截至11月11日，A股融资余额时隔九年，再次突破1.8万亿，仅次于2015年融资余额高点。..."

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_text(text)

# Create documents with metadata (empty dictionary if no metadata)
documents = [Document(page_content=split, metadata={}) for split in splits]

# Initialize the embedding model
embedding = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Create a vector store from the documents
vector_store = Chroma.from_documents(documents=documents, embedding=embedding)

# Create retriever from the vector store
retriever = vector_store.as_retriever()

# Define the system prompt
system_prompt = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.\n
"""

# Create the prompt and chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", f"{input}")
    ]
)

chain1 = create_stuff_documents_chain(model, prompt)
chain2 = create_retrieval_chain(retriever, chain1)

# Invoke the chain to answer a question
resp = chain2.invoke({'input': "What is finance?"})
print(resp)
