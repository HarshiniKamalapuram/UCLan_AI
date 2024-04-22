import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
# from flask import Flask, render_template, request, redirect
# Import necessary libraries
from PyPDF2 import PdfReader
import json

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#
OPENAI_API_KEY = 'api'
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

vectorstore = None
conversation_chain = None
chat_history = []

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Streamlit
# streamlit run pdf_app2.py
def main():
    global text, chunks, vectorstore, conversation_chain, chat_history

    st.title('My UCLan Boat')

  #  pdf_docs = st.file_uploader("Upload PDF", type=['pdf'], accept_multiple_files=True)
    with open(r"About_UCLan.txt", "r", encoding="latin-1") as file:
        raw_text = file.read() 
        #raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)

    user_question = st.text_input("Enter your question:")
    if st.button('Submit'):
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history'] 
        #parsed_json = json.loads(chat_history[1])
        #st.write(parsed_json["AIMessage"])
        st.write(chat_history[1])

if __name__ == '__main__':
    main()

