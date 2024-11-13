import os

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangChainDemo"
os.environ["LANGCHAIN_API_KEY"] = "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"

host = "127.0.0.1"
port = "11434"

# Fin-robot
# 1. Obtain llama3.1:latest model
model = OllamaLLM(base_url=f"http://{host}:{port}", model="llama3.1:latest", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are an optimistic helper, please answer the questions in {language} at your best!'),
    MessagesPlaceholder(variable_name='my_msg')
])

chain = prompt_template | model

# Chat history
# Key: sessionId, Value: chat history object
store = {}


# Accept a seesionId and return the chat history object.
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # The key for sending messages each time
)

# Define a sessionId for current dialogue
config = {'configurable': {'session_id': 'zs123'}}

# Round 1
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好，请问香港中文大学在哪里?')],
        'language': 'english'
    },
    config=config
)
print(resp)

# Round 2
resp = do_message.invoke(
    {
        'my_msg': [HumanMessage(content='你好，请问香港大学在哪里?')],
        'language': 'english'
    },
    config=config
)
print(resp)

# Round 3: Stream data
config = {'configurable': {'session_id': 'lisi1234'}}
for resp in do_message.stream({'my_msg': [HumanMessage(content='你好，请问香港理工大学在哪里?')], 'language': 'english'},
                              config=config):
    print(resp, end='-')
