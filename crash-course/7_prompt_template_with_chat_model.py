from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")


template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic": "cat"})
result = model.invoke(prompt)
print("----- Prompt from Template -----")
print(result.content)


template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_template = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_template.invoke({"adjective": "funny", "animal": "panda"})
result = model.invoke(prompt)
print("----- Prompt with Multiple Placeholders -----")
print(result.content)


messages = [
    ("system", "You are a comedian who tells a jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print("----- Prompt with System and Human Messages (Tuple) -----")
print(result.content)
