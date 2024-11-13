import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = "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"

host = "127.0.0.1"
port = "11434"

# Fin-robot
# 1. Obtain llama3.1:latest model
model = OllamaLLM(base_url=f"http://{host}:{port}", model="llama3.1:latest", temperature=0)

# Prepare data
documents = [
    Document(
        page_content="Financial analysis and investment strategies are crucial for successful stock trading.",
        metadata={"source": "finance_article_1"}
    ),
    Document(
        page_content="Machine learning models can predict stock prices based on historical data trends.",
        metadata={"source": "tech_blog_2"}
    ),
    Document(
        page_content="The global economy is affected by various factors, including inflation and interest rates.",
        metadata={"source": "economics_news_3"}
    ),
    Document(
        page_content="Understanding Python programming can be beneficial for data scientists in finance.",
        metadata={"source": "programming_guide_4"}
    ),
    Document(
        page_content="Portfolio diversification reduces risk by spreading investments across various assets.",
        metadata={"source": "investment_tip_5"}
    ),
    Document(
        page_content="Deep learning techniques are being applied to natural language processing in finance.",
        metadata={"source": "ai_research_6"}
    ),
    Document(
        page_content="Real estate investment involves analyzing property values, location, and market trends.",
        metadata={"source": "real_estate_guide_7"}
    ),
    Document(
        page_content="Cryptocurrencies offer a decentralized digital currency option but come with high volatility.",
        metadata={"source": "crypto_blog_8"}
    ),
    Document(
        page_content="Interest rates have a significant impact on the bond market and overall investment returns.",
        metadata={"source": "bond_market_analysis_9"}
    ),
    Document(
        page_content="Artificial intelligence is transforming the financial industry through automation.",
        metadata={"source": "tech_in_finance_10"}
    )
]

# Initialize a vector database
# 1. 加载本地的 all-MiniLM-L12-v2 嵌入模型
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")


# 2. 创建自定义嵌入类，并继承 Embeddings 接口
class LocalEmbedding(Embeddings):
    def embed_documents(self, texts):
        return embedding_model.encode(texts, convert_to_tensor=True).tolist()

    def embed_query(self, text):
        return embedding_model.encode([text], convert_to_tensor=True).tolist()[0]


# 3. 初始化向量空间
vector_store = Chroma.from_documents(documents, embedding=LocalEmbedding())
# 4. 返回相似度的分数，分数越低相似度越高
# print(vector_store.similarity_search_with_score('stock investment strategies'))

# 检索器 选取相似度最高的第一个
retriever = RunnableLambda(vector_store.similarity_search).bind(k=1)
# print(retriever.batch(['stock', 'investment', 'python']))

# 提示模版
message = """
使用提供的上下文回答这个问题
{question}
上下文：
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([('human', message)])

# RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model
chain={'question': RunnablePassthrough(), 'context': retriever} | prompt_temp | model
print(chain.invoke('please introduce Interest rates'))
