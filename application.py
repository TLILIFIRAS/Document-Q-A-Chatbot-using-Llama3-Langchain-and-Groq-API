import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

load_dotenv()


# Setting Required API Keys 
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

st.title("Document Q&A Chatbot using Llama3,Langchain and Groq API By Eng.Firas Tlili")


llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

prompt1=st.text_input("Enter Your Question :")

def vector_embedding():
    # Function to create vector embeddings for documents
    
    if "vectors" not in st.session_state:
        # Check if the 'vectors' key is already in session state
        # If not, proceed with creating embeddings and storing them in session state
        
        st.session_state.embeddings = OpenAIEmbeddings()
        # Initialize OpenAIEmbeddings to create vector representations of text data
        
        st.session_state.loader = PyPDFDirectoryLoader("./input")  # Data Ingestion
        # Load PDF documents from the specified directory ("./input")
        
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        # Load the documents into session state for further processing
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )  # Chunk Creation
        # Initialize the text splitter to break documents into smaller chunks
        # Chunks will be of size 1000 characters with an overlap of 200 characters
        
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:20]
        )  # Splitting
        # Split the first 20 documents into chunks for better processing and embedding
        
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )  # Vector OpenAI embeddings
        # Create vector embeddings for the final set of document chunks using OpenAIEmbeddings
        # Store the resulting vectors in the session state for later retrieval


if st.button("Documents Embeddings"):
    # Create a button in the Streamlit app with the label "Documents Embeddings"
    # When this button is clicked, the code inside the `if` block will execute

    vector_embedding()
    # Call the vector_embedding() function to create and store vector embeddings in the session state

    st.write("Vector Store DB Is Ready")
    # Display a message in the Streamlit app indicating that the vector store database is ready

if prompt1:
    # Check if the variable 'prompt1' contains a value (i.e., it's not empty or None)
    # If it does, the following code block will execute
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Create a document processing chain using the specified language model (llm) and prompt
    # This chain will be responsible for combining or processing the documents

    retriever = st.session_state.vectors.as_retriever()
    # Convert the stored vector embeddings into a retriever object
    # This retriever will be used to find relevant documents based on the input query

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Create a retrieval chain that ties together the retriever and document chain
    # This allows for querying the documents and getting a response based on relevance

    start = time.process_time()
    # Record the current CPU process time to measure the duration of the retrieval process

    response = retrieval_chain.invoke({'input': prompt1})
    # Invoke the retrieval chain with the input prompt (prompt1)
    # This returns a response that typically contains the answer and related context
    
    print("Response time:", time.process_time() - start)
    # Print the time taken to process the response, showing the elapsed CPU time

    st.write(response['answer'])
    # Display the answer from the response in the Streamlit app
    # With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Create an expandable section in the Streamlit app titled "Document Similarity Search"
        
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # Loop through the context provided in the response, which contains relevant document chunks
            
            st.write(doc.page_content)
            # Display the content of each relevant document chunk
            
            st.write("--------------------------------")
            # Add a separator line between document chunks for better readability

    
    