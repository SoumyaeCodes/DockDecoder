from langchain_groq import ChatGroq 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain.chains.combine_documents import create_stuff_documents_chain


import streamlit as st

import os
import time

import re
import tempfile
import pymupdf as fitz
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="DockDecoder", page_icon="📖")

st.title("DOCUMENT SUMMARIZATION TOOL")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile") 

uploaded_file = st.file_uploader("Upload file here", type="pdf", accept_multiple_files=False)

Prompt = ChatPromptTemplate.from_template(
"""
You are tasked with generating a concise summary of a given text. Your goal is to capture the main ideas, key arguments, and significant details without extraneous information. Please follow these guidelines:

Structure: Present the summary in a logical order that reflects the original text's flow.
Length: Keep the summary brief, ideally within 4-9 sentences.
Clarity: Use clear and straightforward language to convey the essence of the text.
Focus: Highlight the main ideas and include necessary details or examples.
Output: Provide the summary in a clear and organized format.

<context>
{context}
<context>
"""
)

def is_section_title(text):
    stripped_text = text.strip()
    unwanted_keywords = ["REFERENCES", "ACKNOWLEDGEMENTS", "APPENDIX", "FOOTNOTES", "NOTES", "BIBLIOGRAPHY"]
    if len(stripped_text) > 1 and stripped_text[0].isdigit():
        if re.match(r'https?://|doi|arxiv', stripped_text, re.I):
            return False
        if any(keyword in stripped_text.upper() for keyword in unwanted_keywords):
            return False
        if re.match(r'^\d+(\.\d+)*\s+.*', stripped_text):
            return True
    return False

def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(stream=BytesIO(pdf_path.read()), filetype="pdf")
    sections = []
    sections_content = []
    current_section = None
    current_content = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks") 
        for block in blocks:
            text = block[4]  
            if is_section_title(text):
                if current_section:
                    sections.append(current_section.strip())
                    sections_content.append("\n".join(current_content).strip())
                current_section = text.strip()  
                current_content = []
            else:
                if current_section:
                    current_content.append(text)
    if current_section:
        sections.append(current_section.strip())
        sections_content.append("\n".join(current_content).strip())
    for i, heading in enumerate(sections):
        if "conclusion" in heading.lower():  
            sections = sections[:i + 1] 
            break  
    cut_len = len(sections)
    sections_content = sections_content[:cut_len]
    text = sections_content[-1]  
    pos = text.find("REFERENCES")
    if pos != -1:
        text = text[:pos].strip() 
    sections_content[-1] = text
    pos = text.find("Acknowledgments")
    if pos != -1:
        text = text[:pos].strip() 
    sections_content[-1] = text
    return sections, sections_content



if st.button("Upload"):
    if uploaded_file is not None:
        sections, content = extract_sections_from_pdf(uploaded_file)

        for i in range(len(sections)):
            combined_text = str(f"Title: {sections[i]}\nContent: {content[i]}")
            documents = [Document(page_content=combined_text)]
            chain = create_stuff_documents_chain(llm, Prompt)
            response = chain.invoke({"context": documents})
            st.write(sections[i])
            st.write(response)
            st.divider()
