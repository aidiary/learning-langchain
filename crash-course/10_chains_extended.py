from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells a jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# chainに独自の処理を追加できる
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n\n{x}")

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)
