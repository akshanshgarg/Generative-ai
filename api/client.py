import requests
import streamlit as st


def get_llama_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={'input': {'topic': input_text}})
    print(response.json())
    return response.json()['output']

st.title('Langchain Demo With LLAMA2 API')
input_text1=st.text_input("Write a poem on")

if input_text1:
    st.write(get_llama_response(input_text1))