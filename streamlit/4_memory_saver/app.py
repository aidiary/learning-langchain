import asyncio

from dotenv import load_dotenv
from graph import invoke_graph
from langchain_core.messages import AIMessage, HumanMessage

import streamlit as st

# https://github.com/shiv248/Streamlit-x-LangGraph-Cookbooks

load_dotenv()

st.title("StreamLit 🤝 LangGraph")
st.markdown("#### 対話履歴の記憶にMemorySaverを使う")


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

        # streamlitはユーザごとに独立なのでスレッドIDは固定でもＯＫ
        # 同じconfigを渡すことでメモリが維持される
        config = {"configurable": {"thread_id": "1"}}

        # invoke_graphには過去の対話履歴すべてではなく今回のユーザ入力のみを渡す
        # 過去の対話履歴はMemorySaverが保持している
        response = asyncio.run(invoke_graph(prompt, placeholder, config))

        st.session_state.messages.append(AIMessage(response))
