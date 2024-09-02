from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

result = model.invoke("Waht is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
