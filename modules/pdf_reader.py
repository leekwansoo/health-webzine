# modules/pdf_parser.py
import json
from pathlib import Path
import os
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI

model = "gpt-4o-mini"

# CharacterTextSplitter를 사용하여 텍스트를 청크(chunk)로 분할하는 코드
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

def parse_pdf(file_path):
    reader = PdfReader(file_path)
    num_page = len(reader.pages)
    #extract text from each page
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
      
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    return documents


def check_file_exist(dir, file_name):
    file_name = file_name
    file_list = os.listdir(dir)
    file_exist = False
    for file in file_list:
        if file == file_name:
            file_exist = True
            return file_exist
     

def generate_question(text):
    prompt = f"Please generate questions fromthe given {text}/,the questionnare should be in same language as given text"
  
    llm = ChatOpenAI(model= model, temperature = 0.2)
    questions = llm.invoke(prompt)
    return questions

def create_query_file(file_name, text):
    file_name = file_name.split('.')[0]
    query_file = "query/" + f"{file_name}" + "_query.txt"
    with open(query_file, "w", encoding="utf-8") as f:
        f.write(text)
    return query_file

def add_qa_file(file_name, qa_pair):
    #print(file_name)
    file_name = file_name.split('.')[0].split('/')[1]
    qa_file = "qa_pair/" + f"{file_name}" + "_qa.txt"

    # Check if the file exists
    if not os.path.exists(qa_file):
        # If the file doesn't exist, create an empty list
        qa_list = []
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)
    # Read the existing data
    with open(qa_file, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
        # Append the new Q&A pair to the list
    qa_list.append(qa_pair)
    # Write the updated list back to the file
    with open(qa_file, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=4)

    return qa_file

# if query already exist return the query and answer
def check_query_exist(file_name, query):
    # convert file_name to qa_file
    file_name = file_name.split('.')[0].split('/')[1]
    qa_file = f"{file_name}" + "_qa.txt"
    qa_files = os.listdir("qa_pair")
    print(qa_file, qa_files)
    # if file is existing in the qa_files_list then rad the json file
    for file in qa_files:
        print(file)
        if qa_file == file:
            print(file)
            qa_file = "qa_pair/" + f"{qa_file}"
            print(qa_file)
            qa_pair_list = json.loads(Path(qa_file).read_text(encoding= "utf-8"))
            for qa_pair in qa_pair_list:
            
                if query == qa_pair["query"]:
                    print(qa_pair)
                    return qa_pair
    return None
    