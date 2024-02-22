import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

# models
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo-preview"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read PDF

loader = PyPDFLoader("insert-your-pdf-here")
pdfData = loader.load()
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
splitData = text_splitter.split_documents(pdfData)

# End Read PDF

collection_name = "pdf_collection"
local_directory = "pdf_vect_embedding"
persist_directory = os.path.join(os.getcwd(), local_directory)

openai_key=os.environ.get('OPENAI_API_KEY')
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectDB = Chroma.from_documents(splitData,
                      embeddings,
                      collection_name=collection_name,
                      persist_directory=persist_directory
                      )
vectDB.persist()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chatQA = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(openai_api_key=openai_key, max_tokens=2934, 
            temperature=0.2, model_name=GPT_MODEL), 
            vectDB.as_retriever(), 
            memory=memory)

chat_history = []
qry = ""
while qry != 'done':
    qry = input('Question: ')
    if qry != exit:
        response = chatQA({"question": qry, "chat_history": chat_history})
        print("Answer:", response["answer"])
