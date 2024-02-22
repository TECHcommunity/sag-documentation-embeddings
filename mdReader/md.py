import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
import markdown
from scipy import spatial  # for calculating vector similarities for search
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
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from unstructured.partition.md import partition_md
from langchain.text_splitter import MarkdownTextSplitter

# models
EMBEDDING_MODEL = "text-embedding-3-large"
GPT_MODEL = "gpt-4-turbo-preview"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read markdown file

loader = UnstructuredMarkdownLoader("insert-your-markdown-here")
markdownData = loader.load()

headers_to_split_on = [
    ("#", "Header 1"),
     ("##", "Header 2"),
     ("###", "Header 3"),
 ]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs_string = markdownData[0].page_content
splitData = markdown_splitter.split_text(docs_string)

# End Read markdown file

collection_name = "md_collection"
local_directory = "md_vect_embedding"
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
            ChatOpenAI(openai_api_key=openai_key, max_tokens=4095, 
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
