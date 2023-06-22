import streamlit as st
from streamlit_chat import message

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage
from langchain.schema import AIMessage

from dotenv import load_dotenv
load_dotenv()

chat = ChatOpenAI(model_name="gpt-3.5-turbo-0613")

try:
    memory = st.session_state["memory"]
except:
    memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(llm=chat,memory=memory)

st.title("Chatbot in Streamlit")
st.caption("by Masumi Morishige")

text_input = st.text_input("Enter your message")
send_button = st.button("Send")

history = []

if send_button:
    send_button = False
    chain(text_input)

    st.session_state["memory"] = memory

    try:
        history = memory.load_memory_variables({})["history"]
    except Exception as e:
        st.error(e)

for index, chat_message in enumerate(history):
    if type(chat_message) == HumanMessage:
        message(chat_message.content, is_user=True, key=2 * index)
    elif type(chat_message) == AIMessage:
        message(chat_message.content, is_user=False, key=2 * index + 1)