import streamlit as st
import os
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With any files")
    st.header("💬 Chatbot")

    # Path to your pre-existing documents
    doc_paths = [
        "AOH LOTO-ATS 1- Line 2.docx",
        "AOH LOTO-ATS 2- Line 1.docx",
        "AOH LOTO-ATS 3- Line 1.docx",
        "AOH LOTO-ATS 3- Line 2.docx"
    ]

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    process = st.button("Process")
    if process:
        # Read text from documents
        files_text = ""
        for doc_path in doc_paths:
            files_text += get_docx_text(doc_path)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        # create vector stores
        vetorestore = get_vectorstore(text_chunks)
        # create conversation chain
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        st.session_state.conversation = get_conversation_chain(vetorestore, openai_api_key)
        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.text_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)

def get_docx_text(file_path):
    doc = docx.Document(file_path)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_text_chunks(text):
    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(messages.content, is_user=True, key=str(i))
            else:
                st.write(messages.content, key=str(i))
        st.write(f"Total Tokens: {cb.total_tokens}" f", Prompt Tokens: {cb.prompt_tokens}" f", Completion Tokens: {cb.completion_tokens}" f", Total Cost (USD): ${cb.total_cost}")

if __name__ == '__main__':
    main()
