from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

classificatioin_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral or escalate: {feedback}.",
        ),
    ]
)

branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_template | model | StrOutputParser()),
    (lambda x: "negative" in x, negative_feedback_template | model | StrOutputParser()),
    (lambda x: "neutral" in x, neutral_feedback_template | model | StrOutputParser()),
    escalate_feedback_template | model | StrOutputParser(),
)

classification_chain = classificatioin_template | model | StrOutputParser()

chain = classification_chain | branches

# review = (
#     "The product is excellent. I really enjoyed using it and found it very helpful."
# )

# review = (
#     "The product is terrible. It broke after just one use and the quality is very poor."
# )

# review = "The product is okay. It works as expected but nothing exceptional."

review = "I'm not sure about the product yet. Can you tell me about its features?"

result = chain.invoke({"feedback": review})
print(result)
