from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


# Promplt template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpfull assistant please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)


# streamlit framework

st.title('Langchain demo with Llama2 API')
input_text = st.text_input("Search the topic you want")

# llama2 LLM

llm = Ollama(model = "llama2")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))

