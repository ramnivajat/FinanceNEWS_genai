import os
import streamlit as st
import joblib
import time
import validators
import asyncio
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from dotenv import load_dotenv
# Load environment variables from .env
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Function to validate URLs
def validate_url(url):
    return validators.url(url)

# Allow dynamic addition of URLs
urls = []
num_urls = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3)
for i in range(num_urls):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url and validate_url(url):
        urls.append(url)
    elif url:
        st.sidebar.error(f"URL {i+1} is invalid.")

# Progress bar and status messages
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

# Summarization pipeline
summarizer = pipeline("summarization", framework="tf") 

# Load OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.9, max_tokens=1000, api_key=openai_api_key)

# Asynchronous function to process URLs
async def process_urls_async(urls):
    loader = UnstructuredURLLoader(urls=urls)
    status_text.text("Loading data...")
    data = await loader.load_async()
    return data

# Main processing button
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one valid URL.")
    else:
        # Load data asynchronously
        data = asyncio.run(process_urls_async(urls))
        progress_bar.progress(30)
        status_text.text("Data loading complete. Splitting text...")

        # Split data into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=10000,
            chunk_overlap=1000
        )
        docs = text_splitter.split_documents(data)
        progress_bar.progress(60)

        # Display summaries
        st.subheader("Summaries of Articles")
        for doc in docs:
            summary = summarizer(doc['content'], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            st.write(summary)

        # Create embeddings and save to FAISS index
        status_text.text("Building embeddings...")
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        vectorstore_openai.save_local("faiss_index")
        progress_bar.progress(100)
        status_text.text("Embedding vector built successfully.")

# Query input and processing
query = st.text_input("Question: ")
if query:
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)

    # Display answer
    st.header("Answer")
    st.write(result["answer"])

    # Display sources
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)




