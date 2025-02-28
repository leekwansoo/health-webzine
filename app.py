import streamlit as st 
import io
import os
import json 
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from modules.pdf_reader import generate_question, parse_pdf, create_query_file, load_pdf, add_qa_file, check_query_exist
from modules.chromadb import load_documents
from modules.query_handler import query_chroma_db
from graph import search_web
from dotenv import load_dotenv
load_dotenv()

st.title("Webzine for SCL Health")

st.session_state["DOCUMENT"] = []

st.session_state["DOCUMENT"] = os.listdir("uploaded")

doc_list =[st.session_state["DOCUMENT"]]

openai_api_key = st.sidebar.text_input("Enter Your OPEN_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
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
options = st.sidebar.radio("Select an option", ["Upload File", "Query from Uploaded File", "Query By Subjects", "Web Search"])

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
            query_list= generate_question(text)
            query_file = create_query_file(file_name, query_list)
            st.session_state["query_file"].append(query_file)
            for query in query_list:
                st.session_state["query_message"].append(query)
                st.sidebar.markdown(query)
                      
           
         
    else:
        st.sidebar.write("Please upload a PDF and select subject to get started.")
        
            
elif options == "Query from Uploaded File":
    st.header("Query from Uploaded File")
    query_file_list = os.listdir("query")
    selected = st.sidebar.selectbox("Select month to query", query_file_list)
    file_name = f"query/{selected}"
    loader =TextLoader(file_name, encoding = "utf-8")
    documents = loader.load()
    query_list = documents[0].page_content.split("\n")
    # Print the list
    i = 0
    for query in query_list:
        i += 1
        st.markdown(query)
        
        col1, col2 = st.columns([2, 5])  # Create two columns for better layout
        with col1:
            st.write(query)  # Display the query
        with col2:
            if st.button(f"Query", key=f"button_{i}"):  # Add a button with a unique key
                response = query_chroma_db(query)
                if response:
                    qa_pair = {"query": query, "answer": response.content}
                    qa_file = add_qa_file(file_name, qa_pair)
                    st.write(response.content)
                    st.write(f"QA pair is saved in {qa_file}")
        
    #main_query(file_name) 

elif options == "Query By Subjects":  
    st.title("Interactive Query Application")
    st.header("Query By Subject")
    query_file_list = os.listdir("query")
    selected = st.sidebar.selectbox("Select Subject", query_file_list)
    file_name = f"query/{selected}"
    with open(file_name, "r", encoding="utf-8") as f:
        s = f.read()
        st.sidebar.markdown(s)
    
    llm = OpenAI(temperature=0.7)
    prompt = PromptTemplate(
    input_variables=["query"],
    template="Here is the question: {query} \n Please provide an answer."
)
    chain = LLMChain(llm=llm, prompt=prompt)
    # Display each query with its own button
    queries = ["1. MZ 세대에서 고혈압이 급증하는 이유는 무엇인가요?",
                "2. 고혈압 관리에 있어 젊은 세대가 간과하는 점은 무엇인가요?",
                "3. 결핵에 대한 일반적인 오해는 무엇이며, 진실은 무엇인가요?",
                "4. 결핵 환자가 느끼는 사회적 낙인은 어떤 영향을 미치나요?",
                "5. 골다공증 치료 전에 치과 진료를 받아야 하는 이유는 무엇인가요?"]
    for query in queries:
        col1, col2 = st.columns([2, 3])  # Create two columns for better layout
        with col1:
            st.write(query)  # Display the query
        with col2:
            if st.button(f"Query", key=query):  # Add a button with a unique key
                response = chain.run(query=query)  # Query the LLM
                st.write(f"### Response for: {query}")
                st.write(response)
          
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