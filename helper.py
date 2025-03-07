from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_system_prompt import load_system_prompt
import nltk
from nltk import data
from langchain_community.document_loaders import PyPDFLoader
from os.path import join, isdir
from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# Check if corpora are already downloaded
if not isdir(join(data.path[0], 'corpora', 'punkt')):
    nltk.download('punkt')

if not isdir(join(data.path[0], 'corpora', 'averaged_perceptron_tagger')):
    nltk.download('averaged_perceptron_tagger')

load_dotenv()

folder_path= r"dataset"



llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
    # other params...
)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


model_kwargs = {'device': 'cpu'}


vector_db_file_path = "faiss_index_database"


def create_vector_db(folder_path):
    try:
        vectordb = FAISS.load_local(vector_db_file_path, embeddings_model, allow_dangerous_deserialization=True)
        print("Existing vector database loaded.")
    except:
        print("No existing vector database found, creating a new one.")
        vectordb = None

    # Get all files from the input folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Process each file in the folder
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            # If it's a PDF file, load the PDF
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)  # Simple text extraction
                pages = loader.load()  # Synchronously load the PDF content

                # Splitting text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  # chunk size (characters)
                    chunk_overlap=200,  # chunk overlap (characters)
                    add_start_index=True,  # track index in original document
                )
                all_splits = text_splitter.split_documents(pages)
                print(f"Split {file} into {len(all_splits)} sub-documents.")

                if vectordb:
                    # Add new documents to the existing vector database
                    vectordb.add_documents(all_splits)
                # else:
                #     # Create a new FAISS instance if it doesn't exist yet
                #     vectordb = FAISS.from_documents(documents=all_splits, embedding=embeddings_model)
                
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    # Save the updated vector database
    vectordb.save_local(vector_db_file_path)
    print("Vector database saved successfully.")

def create_csv_vector_db(file_path):
    try:
        vectordb = FAISS.load_local(vector_db_file_path, embedding=embeddings_model, allow_dangerous_deserialization=True)
        print("Existing vector database loaded.")
    except:
        print("No existing vector database found, creating a new one.")
        vectordb = None

    # Loop through all provided CSV files and load their data
    # file_path = 'csv_dataset/courses_info.csv'
    loader = CSVLoader(file_path, source_column="Question")
    data = loader.load()

    if vectordb:
        # Add new documents to the existing vector database
        vectordb.add_documents(data)
    else:
        # Create a new FAISS instance if it doesn't exist yet
        print('here i am creating new db')
        vectordb = FAISS.from_documents(documents=data, embedding=embeddings_model)
    
    # Save the updated vector database
    vectordb.save_local(vector_db_file_path)

system_prompt_string = load_system_prompt()

def get_QA_chain():
    
    # Load the vector database from the local folder
    try:
        vectordb = FAISS.load_local(vector_db_file_path, embeddings_model,allow_dangerous_deserialization=True)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(
                        search_type="mmr",
                        # search_type="similarity", 
                        search_kwargs={
                            "score_threshold": 0.2,
                            "k": 10
                            }
                        )
        

        system_prompt = (system_prompt_string)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

    except Exception as e:
        print("Error creating QA chain: ")
        raise


if __name__ == "__main__":
    # create_vector_db(folder_path)
    # create_vector_db(folder_path)
    # create_csv_vector_db('csv_dataset/tbh-bot-qa.csv')
    chain = get_QA_chain()
    print(chain.invoke({"input": "What can you tell me about bridge of hopes?"}))
