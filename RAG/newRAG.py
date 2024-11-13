import mimetypes
import os

import fitz
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style, init
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize colorama for colored console output
init(autoreset=True)

# Set environment variables (if needed)
os.environ['USER_AGENT'] = 'FinTechHelper/1.0 (xzco.work@gmail.com)'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG"
os.environ["LANGCHAIN_API_KEY"] = "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"

llm = ChatOllama(model="llama3.1:latest")


class SentenceTransformersEmbedding(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False).tolist()[0]


def load_data(source):
    # Detect file type
    mime_type, _ = mimetypes.guess_type(source)
    # Text file
    if mime_type == 'text/plain':
        with open(source, 'r', encoding='utf-8') as file:
            return file.read()
    # PDF file
    elif mime_type == 'application/pdf':
        return extract_text_from_pdf(source)
    # Excel/Spreadsheet file
    elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return extract_text_from_excel(source)
    # Webpage (assuming URL input)
    elif mime_type is None and source.startswith("http"):
        return extract_text_from_webpage(source)
    else:
        raise ValueError(f"Unsupported file type for source: {source}")

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    return text

def extract_text_from_excel(file_path):
    df = pd.read_excel(file_path)
    return df.to_string(index=False)

def extract_text_from_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator="\n")




def split_text(text, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.create_documents([text])



# ragData = ["1.txt",
#            "/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/Me/CV of YifeiLi.pdf",
#            "https://csdnnews.blog.csdn.net/article/details/143727894?spm=1000.2115.3001.5927",
#            "/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/CUHK/CMSC5713-IT Project Management/Risk Analysis.xlsx"]
# texts = []
# for source in ragData:
#     raw_text = load_data(source)
#     texts.extend(split_text(raw_text))


def load_all_files_in_directory(directory_path):
    # Supported file extensions for RAG processing
    supported_extensions = (".txt", ".pdf", ".xlsx", ".csv")
    files = []

    # Walk through the directory to find all files with supported extensions
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(supported_extensions):
                files.append(os.path.join(root, filename))
    return files


# Use the function to load all files from your target directory
target_directory = "/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/CUHK/CMSC5720Project/LLM/data"
ragData = load_all_files_in_directory(target_directory)

# Initialize texts list for storing processed data
texts = []

# Loop through each file in ragData and process it
for source in ragData:
    raw_text = load_data(source)  # Assume this function is already defined to handle different file types
    texts.extend(split_text(raw_text))  # Assume this function is defined to split text into chunks



persist_directory = "./vector_store"
embedding_model = SentenceTransformersEmbedding("sentence-transformers/all-MiniLM-L12-v2")

# Load or create the vector store
if os.path.exists(persist_directory):
    print(f"{Fore.YELLOW}Loading existing vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
else:
    print(f"{Fore.YELLOW}Creating new vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
    vector_store.persist()  # Persist the vector store to disk



# 设置查询模板
prompt = PromptTemplate.from_template("""
    Answer the questions using the provided context. If the answer is not in the context, say "cannot find the context". 

    Context: {context}
    Question: {question}
    Answer:
""")

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 测试查询
queries = ["please introduce the newest financial statements of apple"]

# Function to print the question, answer, source documents with similarity scores
def print_answer_and_sources(query, answer, source_documents, query_embedding):
    print(f"\n{Fore.CYAN}Question:{Style.RESET_ALL} {query}")
    print(f"{Fore.YELLOW}Answer:{Style.RESET_ALL} {answer}")
    print(f"{Fore.MAGENTA}Source Documents with Similarity Scores:{Style.RESET_ALL}")

    for doc in source_documents:
        # Accessing the text and calculating similarity score manually
        print(f"{Fore.GREEN}Document:{Style.RESET_ALL} {doc.page_content}")

        # Calculate similarity score between query and document embedding
        doc_embedding = doc.metadata.get('embedding', None)
        if doc_embedding is not None:
            score = np.dot(query_embedding, doc_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        else:
            score = 'N/A'  # If embedding is not available

        print(f"{Fore.RED}Similarity Score:{Style.RESET_ALL} {score}")
    print("-" * 50)


# Loop through queries with customized progress bar
for query in tqdm(
        queries,
        desc=f"{Fore.GREEN}Processing queries{Style.RESET_ALL}",
        bar_format="{l_bar}%s{bar}%s| {n_fmt}/{total_fmt} | Speed: {rate_fmt}" % (Fore.GREEN, Style.RESET_ALL),
        ncols=100,
        unit="it",  # Set the unit to "it" for iteration
        unit_scale=True,  # Enable scaling for better readability if the count is large
):
    # Calculate embedding for the query
    query_embedding = embedding_model.embed_query(query)

    # Retrieve relevant documents
    resp = qa_chain.invoke({"query": query})

    # Add embeddings to each document metadata to calculate similarity
    for doc in resp["source_documents"]:
        doc.metadata['embedding'] = embedding_model.embed_query(doc.page_content)

    # Call the function to print the results in a more structured way
    print_answer_and_sources(query, resp['result'], resp["source_documents"], query_embedding)


