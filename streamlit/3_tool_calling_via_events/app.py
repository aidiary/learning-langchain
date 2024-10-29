import asyncio

from dotenv import load_dotenv
from graph import invoke_graph
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st

# https://github.com/shiv248/Streamlit-x-LangGraph-Cookbooks

load_dotenv()

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### 非同期イベントを使ったストリーム処理とツール呼び出し")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="何かお手伝いできることはありますか？")
    ]

# すべてのメッセージを描画する
for msg in st.session_state.messages:
    if type(msg) is AIMessage:
        st.chat_message("assistant").write(msg.content)
    if type(msg) is HumanMessage:
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.container()
        response = asyncio.run(invoke_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append(AIMessage(response))
