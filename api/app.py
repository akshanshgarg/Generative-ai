from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv


app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description= "A simple API server"
)

add_routes(
    app,
    OllamaLLM(model="llama2"),
    path="/lamma2"
)

llm = OllamaLLM(model="llama2")

prompt1 = ChatPromptTemplate.from_template("Write me a essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")


add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)


