# Retriever and Chain with langchain
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()


os.environ["USER_AGENT"]=os.getenv("USER_AGENT")


# Set the correct file path

attention_file_path = os.path.join(os.getcwd(), 'rag', 'attention.pdf')


# Pdf reader
loader = PyPDFLoader(attention_file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_document = text_splitter.split_documents(docs)

print(split_document[:2])

persist_directory = "./vector_store"

## Vector Embedding And Vector Store
embeddings = OllamaEmbeddings(model='llama3')

# db = Chroma.from_documents(split_document[:15], embeddings, persist_directory=persist_directory)
# db.persist()
# print(f"Vector store saved at {persist_directory}")

db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
print(db)
print("Vector store loaded!")

query = "An attention function can be described as mapping a query "
retireved_results = db.similarity_search(query)
print(retireved_results[0].page_content)

## Load Ollama LAMA2 LLM model
llm=Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>
Question: {input}""")

document_chain=create_stuff_documents_chain(llm,prompt)

"""
Retrievers: A retriever is an interface that returns documents given
 an unstructured query. It is more general than a vector store.
 A retriever does not need to be able to store documents, only to 
 return (or retrieve) them. Vector stores can be used as the backbone
 of a retriever, but there are other types of retrievers as well. 
 https://python.langchain.com/docs/modules/data_connection/retrievers/   
"""

retriever=db.as_retriever()

retrieval_chain=create_retrieval_chain(retriever,document_chain)
response=retrieval_chain.invoke({"input":"Scaled Dot-Product Attention"})
print(response['answer'])