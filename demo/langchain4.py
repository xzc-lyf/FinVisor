import os

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.prebuilt import chat_agent_executor

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"

# 设置主机和端口
host = "127.0.0.1"
port = "11434"

# 初始化模型
llm = OllamaLLM(base_url=f"http://{host}:{port}", model="llama3.1:latest", temperature=0)

# # 定义搜索工具
search = TavilySearchResults(max_results=2)
tools = [search]

agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)

# 4. agent调用
resp = agent_executor.invoke({'messages': [HumanMessage(content='中国首都是哪个城市？')]})
print(resp['messages'])
resp2 = agent_executor.invoke({'messages': [HumanMessage(content='香港天气今天怎么样？')]})
print(resp2['messages'])




# agent.invoke("What is the 25% of 300?")

# agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)
#
# resp = agent_executor.invoke({'messages': [HumanMessage(content='中国首都是哪个城市？')]})
# print(resp['messages'])
# resp2 = agent_executor.invoke({'messages': [HumanMessage(content='香港天气今天怎么样？')]})
# print(resp2['messages'])



