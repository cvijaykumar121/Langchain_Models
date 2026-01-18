from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=30)
result = model.invoke("Write a 3 line poem on Artificial Intelligence")
print(result.content)
