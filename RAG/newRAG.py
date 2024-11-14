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
from PIL import Image
import pytesseract

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
    mime_type, _ = mimetypes.guess_type(source)
    if mime_type == 'text/plain':
        with open(source, 'r', encoding='utf-8') as file:
            return file.read()
    elif mime_type == 'application/pdf':
        return extract_text_from_pdf(source)
    elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return extract_text_from_excel(source)
    elif mime_type and mime_type.startswith('image'):
        return extract_text_from_image(source)
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

def extract_text_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image, lang='chi_sim+eng')  # Use Chinese + English OCR model
    return text

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

def load_all_files_in_directory(directory_path):
    supported_extensions = (".txt", ".pdf", ".xlsx", ".csv", ".jpg", ".png")
    files = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(supported_extensions):
                files.append(os.path.join(root, filename))
    return files

target_directory = "/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/CUHK/CMSC5720Project/LLM/data"
ragData = load_all_files_in_directory(target_directory)

texts = []

for source in ragData:
    raw_text = load_data(source)
    texts.extend(split_text(raw_text))

persist_directory = "./vector_store"
embedding_model = SentenceTransformersEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if os.path.exists(persist_directory):
    print(f"{Fore.YELLOW}Loading existing vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
else:
    print(f"{Fore.YELLOW}Creating new vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
    vector_store.persist()

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

def self_critic_retrieval(query, feedback_threshold=0.7):
    try:
        # Initial answer generation
        response = qa_chain.invoke({"query": query})
        initial_answer = response['result']
        source_docs = response["source_documents"]

        query_embedding = embedding_model.embed_query(query)

        # Self-criticism: calculate similarity between query and each source document
        best_score = 0
        best_docs = []
        for doc in source_docs:
            doc_embedding = embedding_model.embed_query(doc.page_content)
            score = np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
            if score > best_score:
                best_score = score
                best_docs = [doc]

        # If the score is below the threshold, attempt a second retrieval
        if best_score < feedback_threshold:
            print(f"{Fore.YELLOW}Initial answer confidence is low (score: {best_score:.4f}), retrying...{Style.RESET_ALL}")
            retriever.search_kwargs['k'] = 5  # Increase retrieval size
            response = qa_chain.invoke({"query": query})
            adjusted_answer = response['result']
            adjusted_source_docs = response["source_documents"]
            best_docs = adjusted_source_docs if best_score < feedback_threshold else source_docs
            final_answer = adjusted_answer if best_score < feedback_threshold else initial_answer
        else:
            final_answer = initial_answer

        print(f"\n{Fore.CYAN}Question:{Style.RESET_ALL} {query}")
        print(f"{Fore.YELLOW}Final Answer:{Style.RESET_ALL} {final_answer}")
        print(f"{Fore.MAGENTA}Source Documents:{Style.RESET_ALL}")
        print_answer_and_sources(query, final_answer, best_docs, query_embedding)

        return final_answer
    except Exception as e:
        print(f"{Fore.RED}Error occurred during retrieval: {str(e)}{Style.RESET_ALL}")
        return "Error occurred during retrieval"

def print_answer_and_sources(query, answer, source_documents, query_embedding):
    print(f"\n{Fore.CYAN}Question:{Style.RESET_ALL} {query}")
    print(f"{Fore.YELLOW}Answer:{Style.RESET_ALL} {answer}")
    print(f"{Fore.MAGENTA}Source Documents with Similarity Scores:{Style.RESET_ALL}")
    if source_documents:
        for doc in source_documents:
            doc_content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            doc_embedding = doc.metadata.get('embedding', None)
            score = 'N/A' if doc_embedding is None else np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))

            print(f"{Fore.GREEN}Document Excerpt:{Style.RESET_ALL} {doc_content}")
            print(f"{Fore.RED}Similarity Score:{Style.RESET_ALL} {score:.4f}" if score != 'N/A' else f"{Fore.RED}Similarity Score:{Style.RESET_ALL} {score}")
            print("-" * 50)
    else:
        print(f"{Fore.RED}No relevant source documents found!{Style.RESET_ALL}")


queries = ["介绍一下结合超图卷积网络和图注意力网络的会话推荐模型"]

for query in tqdm(
        queries,
        desc=f"{Fore.GREEN}Processing queries{Style.RESET_ALL}",
        bar_format="{l_bar}%s{bar}%s| {n_fmt}/{total_fmt} | Speed: {rate_fmt}" % (Fore.GREEN, Style.RESET_ALL),
        ncols=100,
        unit="it",
        unit_scale=True,
):
    try:
        query_embedding = embedding_model.embed_query(query)
        resp = qa_chain.invoke({"query": query})
        for doc in resp["source_documents"]:
            doc.metadata['embedding'] = embedding_model.embed_query(doc.page_content)

        # Display concise answer and sources
        print_answer_and_sources(query, resp['result'], resp["source_documents"], query_embedding)

        # Optionally, you can call self-critic retrieval if needed
        final_answer = self_critic_retrieval(query)
    except Exception as e:
        print(f"{Fore.RED}Error processing query: {str(e)}{Style.RESET_ALL}")
