import os
import requests
import streamlit as st
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Load API key from environment variables
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
            else:
                st.warning(f"No text found on page {pdf_reader.pages.index(page) + 1} of {pdf.name}")
    if not text:
        st.error(f"No text extracted from the provided PDF files.")
    return text

def get_url_text(urls):
    text = ""
    for url in urls:
        if url.strip() == "":
            continue
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    text += response.text
                else:
                    st.warning(f"URL {url} did not return text/html content. Content-Type: {content_type}")
            else:
                st.warning(f"Failed to fetch URL {url}. Status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error fetching URL {url}: {e}")
    if not text:
        st.error("No text extracted from the provided URLs.")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    docs = [Document(page_content=chunk) for chunk in text_chunks]

    if not docs:
        raise ValueError("No documents to index. Make sure there is content in the PDF or URLs.")
    
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    vectorstore_openai.save_local("faiss_index")



def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo-instruct', temperature=0.6, max_tokens=1000)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=new_db.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)

    st.write("Reply: ", result["answer"])
    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)

def main():
    st.set_page_config(page_title="Chat PDF and URL")
    st.header("News Research Tool 📈")

    user_question = st.text_input("Ask a Question from the given Files")

    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Article URL and PDF")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}")
            urls.append(url)
        
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                pdf_text = get_pdf_text(pdf_docs) if pdf_docs else ""
                url_text = get_url_text(urls) if any(urls) else ""
                
                st.write(f"Extracted PDF text length: {len(pdf_text)}")
                st.write(f"Extracted URL text length: {len(url_text)}")
                
                if not pdf_text and not url_text:
                    st.error("No text found in the provided PDFs or URLs.")
                    return

                raw_text = pdf_text + url_text
                text_chunks = get_text_chunks(raw_text)
                
                if not text_chunks:
                    st.error("No text chunks generated from the provided content.")
                    return
                
                get_vector_store(text_chunks)
                st.success("Done")   

if __name__ == "__main__":
    main()
