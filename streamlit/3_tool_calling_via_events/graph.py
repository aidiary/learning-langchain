from typing import Annotated, Literal, TypedDict

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import StructuredTool, tool
from langchain_openai import ChatOpenAI

import streamlit as st
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

# 検索ツールの定義
search_ddg = StructuredTool.from_function(
    name="Search",
    func=DuckDuckGoSearchAPIWrapper().run,
    description="時事問題についての質問に答える必要があるときに役立つ。的を絞った質問をすべき",
)


@tool
def get_weather(location: str):
    """現在の天気を取得する"""
    if location in ["東京", "Tokyo"]:
        return "晴れです"
    else:
        return "曇っています"


@tool
def get_coolest_cities():
    """最もクールな都市を取得する"""
    return "東京, サンフランシスコ"


tools = [get_weather, get_coolest_cities, search_ddg]
tool_node = ToolNode(tools)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph = StateGraph(GraphState)


def should_continue(state: GraphState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


def _call_model(state: GraphState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.7,
        streaming=True,
    ).bind_tools(tools, parallel_tool_calls=False)
    response = llm.invoke(messages)
    return {"messages": [response]}


graph.add_node("model_node", _call_model)
graph.add_node("tools", tool_node)

graph.add_edge(START, "model_node")
graph.add_conditional_edges("model_node", should_continue)
graph.add_edge("tools", "model_node")

graph_runnable = graph.compile()


async def invoke_graph(st_messages, st_placeholder):
    container = st_placeholder
    thoughts_placeholder = container.container()
    token_placeholder = container.empty()
    final_text = ""

    async for event in graph_runnable.astream_events(
        {"messages": st_messages}, version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            addition = event["data"]["chunk"].content
            final_text += addition
            if addition:
                token_placeholder.write(final_text)
        elif kind == "on_tool_start":
            with thoughts_placeholder:
                status_placeholder = st.empty()
                with status_placeholder.status("Calling Tool...", expanded=True) as s:
                    st.write("Called ", event["name"])
                    st.write("Tool input: ")
                    st.code(event["data"].get("input"))
                    st.write("Tool output: ")
                    output_placeholder = st.empty()
                    s.update(label="Completed Calling Tool!", expanded=False)
        elif kind == "on_tool_end":
            with thoughts_placeholder:
                if "output_placeholder" in locals():
                    output_placeholder.code(event["data"].get("output").content)

    return final_text
