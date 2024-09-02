from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells a jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(response)
