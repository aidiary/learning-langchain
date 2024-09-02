from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI

PROJECT_ID = "langchain-demo-61266"
SESSION_ID = "user_sessioin_1"
COLLECTION_NAME = "chat_history"

print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# 既存のセッションがあれば会話履歴を復元できる
print("Initializing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID, collection=COLLECTION_NAME, client=client
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

model = ChatOpenAI(model="gpt-4o-mini")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
