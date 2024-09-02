from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="what is 81 divided by 9?"),
]

model = ChatOpenAI(model="gpt-4o-mini")
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")

model = ChatAnthropic(model="claude-3-haiku-20240307")
result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
