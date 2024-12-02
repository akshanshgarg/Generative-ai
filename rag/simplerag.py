## Data Ingestion
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


os.environ["USER_AGENT"]=os.getenv("USER_AGENT")


# Set the correct file path

speech_file_path = os.path.join(os.getcwd(), 'rag', 'speech.txt')
attention_file_path = os.path.join(os.getcwd(), 'rag', 'attention.pdf')


try:
    loader = TextLoader(speech_file_path)
    documents = loader.load()
except RuntimeError as e:
    print(f"Error loading the file: {e}")


# Web based loader
loader2 = WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                      bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                      class_ = ("post-title","post-content","post-header")
                      )))
text_documents2 = loader2.load()

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

query = "Who are the authors of attention is all you need"
retireved_results = db.similarity_search(query)
print(retireved_results[0].page_content)
