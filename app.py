from langchain_groq import ChatGroq 
from langchain_community.vectorstores import FAISS #vectorStore DB
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings #vector embedding technique
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain


import streamlit as st

import os
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")


st.title("DOCUMENT SUMMARIZATION TOOL")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile") 

uploaded_file = st.file_uploader("Please Upload the PDF file you want to SUMMARIZE", type="pdf", accept_multiple_files=False)

Prompt = ChatPromptTemplate.from_template(
"""
You are tasked with extracting titles and subtitles from a PDF document. Your goal is to read the document and provide a list of titles and subtitles as they appear in the text. Please follow these guidelines:

Structure: Present the titles and subtitles in the order they occur in the document.
Format: Clearly distinguish between titles and subtitles:
Use "Title:" for titles.
Use "Subtitle:" for subtitles.
Context: If a title has multiple subtitles, list them underneath the corresponding title.
Page Numbers: Indicate the start and end page numbers for each title and subtitle pair, formatted as (Page X - Page Y).
Output: Provide the extracted information in a clear and organized format.

<context>
{context}
<context>
"""
)


def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        if uploaded_file:
            temppdf=f"temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFLoader(temppdf)
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)


if st.button("Upload"):
    vector_embedding(uploaded_file)
    docs =[]
    for doc in st.session_state.final_documents:
        docs.append(doc)
    document_chain = create_stuff_documents_chain(llm, Prompt)
    start = time.process_time()
    response = document_chain.invoke({"context": docs})
    st.write(response)







