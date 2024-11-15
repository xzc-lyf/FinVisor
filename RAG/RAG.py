import logging
import mimetypes
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import fitz
import numpy as np
import openpyxl
import pandas as pd
import requests
from bs4 import BeautifulSoup
from colorama import Fore, Style, init
from easyocr import easyocr
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from torch.compiler import disable
from tqdm import tqdm
from PIL import Image, ImageFilter
import pytesseract

# Initialize colorama for colored console output
init(autoreset=True)

# Set environment variables
os.environ['USER_AGENT'] = 'FinTechHelper/1.0 (xzco.work@gmail.com)'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG"
os.environ["LANGCHAIN_API_KEY"] = "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"

COLOR_GREEN = Fore.GREEN
COLOR_RED = Fore.RED
COLOR_CYAN = Fore.CYAN
COLOR_YELLOW = Fore.YELLOW
COLOR_MAGENTA = Fore.MAGENTA
COLOR_WHITE = Fore.WHITE
RESET = Style.RESET_ALL


class SentenceTransformersEmbedding(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False, show_progress_bar=True).tolist()[0]


def load_and_process_files(files_path_lists):
    texts = []
    for file_path in files_path_lists:
        raw_text = load_text_from_file(file_path)
        cleaned_text = clean_text(raw_text)
        texts.extend(split_text(cleaned_text))
    return texts


def initialize_vector_store(texts, embedding_model, persist_directory):
    if os.path.exists(persist_directory):
        logging.info(f"{COLOR_YELLOW}Loading existing vector store...{RESET}")
    else:
        logging.info(f"{COLOR_YELLOW}Creating new vector store...{RESET}")
    return Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)


def attain_paths_of_all_files(directory_path):
    supported_extensions = (".txt", ".pdf", ".xlsx", ".csv", ".jpg", ".png")
    files_paths = []
    for root, dirs, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(supported_extensions):
                files_paths.append(os.path.join(root, filename))
    return files_paths


def load_text_from_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'text/plain':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif mime_type == 'application/pdf':
        return extract_text_from_pdf(file_path)
    elif mime_type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
        return extract_text_from_excel(file_path)
    elif mime_type and mime_type.startswith('image'):
        return extract_text_from_image(file_path)
    elif mime_type is None and file_path.startswith("http"):
        return extract_text_from_webpage(file_path)
    else:
        raise ValueError(f"Unsupported file type for source: {file_path}")


def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as pdf:
        return "".join([page.get_text() for page in pdf])


def extract_text_from_excel(file_path):
    wb = openpyxl.load_workbook(file_path)
    all_text = []
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        sheet_text = f"Sheet: {sheet_name}\n"
        for row in sheet.iter_rows(values_only=True):
            row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
            sheet_text += row_text + "\n"
        all_text.append(sheet_text)
    return "\n".join(all_text)


def extract_text_from_image(file_path):
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    results = reader.readtext(file_path)
    extracted_text = "\n".join([text for _, text, _ in results])
    return extracted_text


def extract_text_from_webpage(url):
    response = requests.get(url)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text(separator="\n")


def split_text(text, chunk_size=1000, chunk_overlap=150):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.create_documents([text])


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    return text


def calculate_similarity(query_embedding, doc_embedding):
    if doc_embedding is None:
        return 'N/A'
    return np.dot(query_embedding, doc_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))


def print_answer_and_sources(query, answer, source_documents, query_embedding):
    print(f"\n{COLOR_CYAN}Question:{RESET}\n{COLOR_WHITE}{query}{RESET}")
    print(f"\n{COLOR_YELLOW}Answer:{RESET}\n{COLOR_WHITE}{answer}{RESET}")
    print(f"\n{COLOR_MAGENTA}Source Documents with Similarity Scores:{RESET}\n")
    print(f"{'Document Excerpt':<50} {'Similarity Score':<20}")

    print("-" * 70)
    for doc in source_documents:
        doc_content = doc.page_content[:20] + "..." if len(doc.page_content) > 20 else doc.page_content
        doc_embedding = doc.metadata.get('embedding', None)
        score = calculate_similarity(query_embedding, doc_embedding)
        score_display = score if score == 'N/A' else f'{score:.4f}'
        print(f"{doc_content:<50} {score_display:<20}")
    print("-" * 70)


def find_best_documents(query_embedding, docs):
    best_score = 0
    best_docs = []
    for doc in docs:
        doc_embedding = doc.metadata.get('embedding')
        score = calculate_similarity(query_embedding, doc_embedding)
        if score > best_score:
            best_score = score
            best_docs = [doc]
    return best_score, best_docs


def self_critic_retrieval(retriever, qa_chain, query, initial_docs, query_embedding, feedback_threshold=0.2):
    best_score, best_docs = find_best_documents(query_embedding, initial_docs)

    if best_score < feedback_threshold:
        print(f"{COLOR_YELLOW}Initial answer confidence is low (score: {best_score:.4f}), retrying...{RESET}")
        retriever.search_kwargs['k'] = 5
        response = qa_chain.invoke({"query": query})
        adjusted_answer = response['result']
        adjusted_source_docs = response["source_documents"]
        best_docs = adjusted_source_docs if best_score < feedback_threshold else initial_docs
        final_answer = adjusted_answer if best_score < feedback_threshold else response['result']
    else:
        final_answer = initial_docs[0].metadata['answer']

    print(f"\n{COLOR_CYAN}Question:{RESET} {query}")
    print(f"{COLOR_YELLOW}Final Answer:{RESET} {final_answer}")
    print(f"{COLOR_MAGENTA}Source Documents:{RESET}")
    print_answer_and_sources(query, final_answer, best_docs, query_embedding)

    return final_answer


def process_query(qa_chain, embedding_model, retriever, query):
    try:
        response = qa_chain.invoke({"query": query})
        source_documents = response["source_documents"]
        query_embedding = embedding_model.embed_query(query)

        for doc in source_documents:
            if 'embedding' not in doc.metadata:
                doc.metadata['embedding'] = embedding_model.embed_query(doc.page_content)

        print_answer_and_sources(query, response['result'], source_documents, query_embedding)

        final_answer = self_critic_retrieval(retriever, qa_chain, query, source_documents, query_embedding)
        return final_answer

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None


def main():
    llm = ChatOllama(model="llama3.1:latest")
    files_path_lists = attain_paths_of_all_files("/Users/xzc/Library/Mobile Documents/com~apple~CloudDocs/CUHK/CMSC5720Project/LLM/data")
    texts = load_and_process_files(files_path_lists)

    persist_directory = "./vector_store"
    embedding_model = SentenceTransformersEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vector_store = initialize_vector_store(texts, embedding_model, persist_directory)
    # vector_store = FAISS.from_documents(texts, embedding_model)

    # prompt = """
    #     Answer the questions using the provided context. If the answer is not in the context, say "cannot find the context".
    #     Context: {context}
    #     Question: {question}
    #     Answer:
    # """

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    queries = ["分析一下苹果财报", "分析中国银行股份有限公司2024年第三季度报告"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        for _ in tqdm(executor.map(partial(process_query, qa_chain, embedding_model, retriever), queries), desc="Processing queries"):
            pass

if __name__ == "__main__":
    main()
