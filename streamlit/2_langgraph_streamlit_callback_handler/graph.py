from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


graph = StateGraph(GraphState)


def _call_model(state: GraphState):
    messages = state["messages"]
    llm = ChatOpenAI(
        temperature=0.0,
        streaming=True,
    )
    response = llm.invoke(messages)

    # LangGraphのReducerによって状態のmessagesに追加される
    return {"messages": [response]}


graph.add_node("modelNode", _call_model)

graph.add_edge(START, "modelNode")
graph.add_edge("modelNode", END)

graph_runnable = graph.compile()


def invoke_graph(st_messages, callables):
    if not isinstance(callables, list):
        raise TypeError("callables must be a list")

    return graph_runnable.invoke(
        {"messages": st_messages}, config={"callbacks": callables}
    )
