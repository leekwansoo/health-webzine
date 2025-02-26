import streamlit as st 
import io
import os
import json 
from modules.pdf_reader import generate_question, parse_pdf, create_query_file, load_pdf, add_qa_file, check_query_exist
from modules.chromadb import load_documents
from modules.query_handler import query_chroma_db
from graph import search_web
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
st.title("Webzine for SCL Health")

st.session_state["DOCUMENT"] = []

st.session_state["DOCUMENT"] = os.listdir("uploaded")

doc_list =[st.session_state["DOCUMENT"]]
    
# Function to add data to DOCUMENT directory 
def check_document(value): 
    if value not in st.session_state["DOCUMENT"]:
        result = "noexist"
        return(result)
    else:
        result = "exist"
        return(result)
       
def add_document(value):         
    st.session_state["DOCUMENT"].append(value)
    st.write(f"Document added: {value}") 

# Example usage # List Document
def list_documents(): 
    if st.session_state["DOCUMENT"]:
        docs = st.session_state["DOCUMENT"]
        return(docs)
    else: st.write("No documents found in 'DOCUMENT'.")

# Main Page content 
def main_query(file_name):
    query = st.text_input("Enter your question for your uploaded documents:") 
    if query: 
        # check if query is in the file
        response = check_query_exist(file_name, query)
        if response:
            st.write(response["answer"])
            #print("same question")
        else:
            response = query_chroma_db(query)
            if response:
                qa_pair = {"query": query, "answer": response.content}
                qa_file = add_qa_file(file_name, qa_pair)
                st.write(response.content)
                st.write(f"QA pair is saved in {qa_file}")
        
st.session_state["query_message"] = []
st.session_state["query_file"] = []

# Create a sidebar for navigation
st.sidebar.title("Menu")
options = st.sidebar.radio("Select an option", ["Upload File", "Query from Uploaded File", "Web Search"])

if options == "Upload File":
    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf", key="pdf_uploader")
       
    if uploaded_file:
        file_name = uploaded_file.name
        check_exist = check_document(file_name)
        check_exist = "noexist"
        if check_exist == "noexist":
            # store the file in the uploaded file folder
            uploaded_name = f"uploaded/{file_name}"
            with open(uploaded_name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            text = parse_pdf(uploaded_file)
            documents = load_pdf(uploaded_name)
            result = load_documents(documents)  # load documents to chromadb
            if result:
                st.sidebar.write(result)
                add_document(file_name)     
            else: st.sidebar.write("storing PDF file into vector store failed")
            questions= generate_question(text)
            query_file = create_query_file(file_name, questions.content)
            st.session_state["query_file"].append(query_file)
            for question in questions.content:
                st.session_state["query_message"].append(question)
            st.sidebar.markdown(questions.content)
                      
            docs = list_documents()
            if docs:
                for doc in docs:
                    st.sidebar.write(f"Uploaded_Document: {doc}\n")  
        else:
            st.sidebar.write(f"{file_name} is aleardy uploaded\n")     
         
    else:
        st.sidebar.write("Please upload a PDF and select subject to get started.")
        
            
elif options == "Query from Uploaded File":
    st.header("Query from Uploaded File")
    query_list = os.listdir("query")
    selected = st.sidebar.selectbox("Select month to query", query_list)
    file_name = f"query/{selected}"
    with open(file_name, "r", encoding="utf-8") as f:
        s = f.read()
        st.sidebar.markdown(s)
        
    main_query(file_name) 
          
elif options == "Web Search":
    st.header("Web Search")
    query = st.text_input("Enter a search query:")
    if st.button("Search Web"):
        if query:
            results = search_web(query)
            for result in results:
                st.write(result["content"])
        else:
            st.write("Please enter a search query.")