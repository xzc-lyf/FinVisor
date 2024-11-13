import os

from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langserve import add_routes

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

host = "127.0.0.1"
port = "11434"

# 1. Obtain llama3.1:latest model
model = OllamaLLM(base_url=f"http://{host}:{port}", model="llama3.1:latest", temperature=0)

# 2. Prepare prompt
msg = [
    SystemMessage(content = '将以下内容翻译为意大利语'),
    HumanMessage(content = '你好，请问你要去哪里？')
]

prompt_template = ChatPromptTemplate.from_messages([
    ('system', '请将下面的内容翻译为{language}'),
    ('user', "{text}")
])

chain = prompt_template | model
print(chain.invoke({'language': 'english', 'text': '我明天要参加一场面试，不能去上课了'}))


# FastAPI
app = FastAPI(title = 'My LangChain Server', version = 'V1.0', description = 'Utilize LangChain to translate sentences')
add_routes(
    app,
    chain,
    path = '/chainDemo'
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "127.0.0.1", port = 8000)