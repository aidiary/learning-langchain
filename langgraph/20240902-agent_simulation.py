from typing import Annotated

from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class SimulatedUser:
    def __init__(self) -> None:
        self.system_prompt_template = """あなたは航空会社の顧客です。
        あなたはカスタマーサポート担当者とやり取りしています。

        {instructions}

        会話が終了したら、単語「FINISHED」で応答してください。"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.name = "Harrison"
        self.instructions = """あなたの名前は Harrison です。あなたはアラスカへの旅行の払い戻しを求めています。
        すべてのお金を返してもらいたいと思っています。
        この旅行は5年前に行われました。"""

        self.prompt = self.prompt.partial(
            name=self.name, instructions=self.instructions
        )

        self.model = ChatOpenAI(model="gpt-4o-mini")

        self.chain = self.prompt | self.model | StrOutputParser()

    def invoke(self, state):
        messages = state["messages"]

        # 役割を入れ替える
        new_messages = self._swap_roles(messages)

        response = self.chain.invoke({"messages": new_messages})

        return {"messages": [HumanMessage(content=response)]}

    def should_continue(self, state):
        messages = state["messages"]
        if len(messages) > 6:
            return "end"
        elif messages[-1].content == "FINISHED":
            return "end"
        else:
            return "continue"

    def _swap_roles(self, messages):
        new_messages = []
        for m in messages:
            if isinstance(m, AIMessage):
                new_messages.append(HumanMessage(content=m.content))
            else:
                new_messages.append(AIMessage(content=m.content))
        return new_messages


class Chatbot:
    def __init__(self) -> None:
        self.system_prompt_template = (
            """あなたは航空会社のカスタマーサポートエージェントです。"""
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        self.model = ChatOpenAI(model="gpt-4o-mini")

        self.chain = self.prompt | self.model | StrOutputParser()

    def invoke(self, state):
        messages = state["messages"]
        response = self.chain.invoke({"messages": messages})
        return {"messages": [AIMessage(content=response)]}


class State(TypedDict):
    messages: Annotated[list, add_messages]


def session(user, chatbot):
    graph_builder = StateGraph(State)
    graph_builder.add_node("user", user.invoke)
    graph_builder.add_node("chatbot", chatbot.invoke)
    graph_builder.add_edge("chatbot", "user")
    graph_builder.add_conditional_edges(
        "user", user.should_continue, {"end": END, "continue": "chatbot"}
    )
    graph_builder.add_edge(START, "chatbot")
    simulation = graph_builder.compile()

    for chunk in simulation.stream({"messages": []}):
        if END not in chunk:
            print(chunk)
            print("-----")


if __name__ == "__main__":
    user = SimulatedUser()
    chatbot = Chatbot()

    session(user, chatbot)
