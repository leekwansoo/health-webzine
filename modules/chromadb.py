from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()
import os
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# Initialize the client without persistent storage
collection_name = "scl_helath_collection"
embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
client = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"  # Directory to save data locally
)

# add documents to the vectorstore

def add_documents(documents, ids):
    result = client.add_documents(documents = documents, ids =ids)
    # return stored ids of documents
    return result

# Manage Vector store
def load_documents(documents):
    # load documents to the vectorstore
    ids = []
    for document in documents:
        document_id = str(uuid4())
        document = Document(
            document_id=document_id,
            page_content=document.page_content,
            metadata=document.metadata,
        )
        ids.append(document_id)
    #print(documents, ids)
    add_documents(documents=documents, ids=ids)
    return "Documents loaded to chromadb successfully"

# delete documents with document ids[]
def delete_documents(ids):
    result = client.delete(ids = ids)
    return result

# search for documents with query
def search_documents(query, k):
    result = client.similarity_search(query = query, k = k)
    return result
    # return document ids[], page_content[], metadata[]
    