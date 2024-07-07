import os
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

from dotenv import load_dotenv
os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_url_text(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text = "\n".join([doc.page_content for doc in data])
    return text

def get_text_chunks(text):
    # Reduce chunk size to avoid exceeding the token limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    docs = [Document(page_content=chunk) for chunk in text_chunks]
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    vectorstore_openai.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), model='gpt-3.5-turbo-instruct', temperature=0.7, max_tokens=100000)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    # Ensure context length does not exceed token limit
    context = ""
    for doc in docs:
        if len(context) + len(doc.page_content) <= 3000:  # Adjust this value as needed
            context += doc.page_content + "\n"

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": [Document(page_content=context)], "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

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
                raw_text = pdf_text + url_text
                
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")   

if __name__ == "__main__":
    main()

