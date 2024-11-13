import os

import numpy as np
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
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = "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"

# Initialize the model with the updated class
llm = ChatOllama(model="llama3.1:latest")

# Load the text from the file
with open("RAG/1.txt", encoding='utf-8') as f:
    last_question = f.read()

# Remove the 'separator' argument since it caused an error
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

# Split the document into chunks
texts = text_splitter.create_documents([last_question])


class SentenceTransformersEmbedding(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False).tolist()[0]


# Define the directory for persistent storage
persist_directory = "./vector_store"
embedding_model=SentenceTransformersEmbedding("sentence-transformers/all-MiniLM-L12-v2")

# Load or create the vector store
if os.path.exists(persist_directory):
    print(f"{Fore.YELLOW}Loading existing vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
else:
    print(f"{Fore.YELLOW}Creating new vector store...{Style.RESET_ALL}")
    vector_store = Chroma.from_documents(texts, embedding=embedding_model, persist_directory=persist_directory)
    vector_store.persist()  # Persist the vector store to disk

# Set up the retriever
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt template
prompt = PromptTemplate.from_template("""
    Answer the questions using the provided context, if the answer is not contained in the context, say "cannot find the context" \n\n
    Context: {context}
    Question: {question}
    Answer:
""")

# Set up the QA chain with RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# List of queries to process
queries = ["你好，今天香港的天气怎么样？"]

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
        print(f"Document embedding: {doc.metadata['embedding']}")


    # Call the function to print the results in a more structured way
    print_answer_and_sources(query, resp['result'], resp["source_documents"], query_embedding)


