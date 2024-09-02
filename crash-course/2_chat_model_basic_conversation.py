from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
    AIMessage(content="81 dvided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")
