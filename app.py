# -------------------------------------------
# Author: Marcos Díaz
# -------------------------------------------
""" Talk to your data webApp, using GPT-3 to interact with own documents.
Includes chat history for conversation memory. """
# -------------------------------------------

import os
import tempfile
from typing import List

import streamlit as st
import tiktoken
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PagedPDFSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import Chroma
from openai.error import APIConnectionError, InvalidRequestError, RateLimitError

st.set_page_config(
    page_title="Talk to your data",
    page_icon="book",
    layout="wide",
    initial_sidebar_state="auto",
)


def load_pdf_files(dir_path: str) -> List[list]:
    """Loads .pdf files in a given directory and extracts its content.
    Args:
        dir_path (str): directory path.
    Returns:
        list: text read per .pdf file in the given directory.
    """
    
    pdf_files_pages = []
    for file in os.listdir(dir_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(dir_path, file)
            loader = PagedPDFSplitter(file_path)
            pages = loader.load_and_split()
            pdf_files_pages.append(pages)
    return pdf_files_pages


def embedding_process(openai_api_key: str, pages: list) -> Chroma:
    """Converts text into vectors.
    Args:
        openai_api_key (str): OpenAI API key.
        pages (list): texts extracted from .pdf files.
    Returns:
        Chroma: vectorised texts.
    """
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = Chroma.from_documents(pages, embeddings)
    return vectorstore


def define_template() -> ChatPromptTemplate:
    """Defines the role for the GPT bot, as well as the model prompt template.
    Returns:
        ChatPromptTemplate: role and prompt template for model interaction.
    """
    
    system_template = """Use the following pieces of context to answer.
    You're an expert in the given context and just in that context. If you're asked something
    which is not indicated in the context, be brief and say you don't know. Do not try to make up an answer
    even if you have the information in your general knowledge.
    ----------------
    {context}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def build_bot_engine(
    openai_api_key: str, prompt: ChatPromptTemplate, vectorstore: object
) -> ChatVectorDBChain:
    """Builds the GPT bot.
    Args:
        openai_api_key (str): OpenAI API key.
        prompt (ChatPromptTemplate): role and prompt template for model interaction.
        vectorstore (object): vectorised texts.
    Returns:
        ChatVectorDBChain: GPT bot engine ready for interaction.
    """
    
    bot_engine = ChatVectorDBChain.from_llm(
        ChatOpenAI(openai_api_key=openai_api_key, max_tokens=1000),
        vectorstore,
        qa_prompt=prompt,
    )
    return bot_engine


def get_bot_reply(chat_history: list, user_input: str, bot_engine: object) -> str:
    """Gets the bot reply given an user question.
    Args:
        chat_history (list): chat conversation history.
        user_input (str): user question.
        bot_engine (object): GPT bot engine ready for interaction.
    Returns:
        str: bot answer.
    """
    
    return bot_engine({"question": user_input, "chat_history": chat_history})


def save_file(filename: str, upload_path: str, file_object: object):
    """Saves a file in local disk.
    Args:
        filename (str): file name.
        upload_path (str): local directory to save the file.
        file_object (object): file object given by the user.
    """
    
    output_file = open(os.path.join(upload_path, filename), "wb")
    output_file.write(file_object)
    
    
def submit():
    st.session_state.user_question = st.session_state.widget
    st.session_state.widget = ""
    
    
def disable():
    st.session_state["disabled"] = True
    
    
def get_num_tokens(text: str) -> int:
    """Calculate num tokens with tiktoken package."""
    
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokenized_text = enc.encode(text)
    return len(tokenized_text)


st.image("datalab_header.PNG", use_column_width=True)
st.title("Talk to your data")
st.caption("🎙 Now you can interact with your own documents!")
st.markdown(" ")

if "disabled" not in st.session_state:
    st.session_state["disabled"] = False
    
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
    
openai_api_key = st.text_input(
    "Indicate your OpenAI API key",
    disabled=st.session_state.disabled,
    on_change=disable,
    type="password",
)

uploaded_file = st.file_uploader(
    "Upload your document(s)", type=["PDF"], accept_multiple_files=True
)


if (
    openai_api_key
    and ("uploaded_file" not in st.session_state)
    and (uploaded_file is not None)
):
    st.session_state["openai_api_key"] = openai_api_key
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with st.spinner(f"Processing Document ..."):
            for file in uploaded_file:
                save_file(file.name, temp_dir, file.getvalue())
                
            pdf_files_pages = load_pdf_files(temp_dir)
            flatten_pages = [page for pdf_file in pdf_files_pages for page in pdf_file]
            
            try:
                docsearch = embedding_process(
                    st.session_state["openai_api_key"], flatten_pages
                )
                prompt = define_template()
                bot_engine = build_bot_engine(
                    st.session_state["openai_api_key"], prompt, docsearch
                )
                
                st.session_state["uploaded_file"] = uploaded_file
                
                if "bot_engine" not in st.session_state:
                    st.session_state["bot_engine"] = bot_engine
                    
            except Exception as error:
                st.warning("Please, upload your PDF document(s).")
                
if "uploaded_file" in st.session_state:
    st.success("**The bot is ready for conversation**")
    st.markdown("---")
    
if "user_question" not in st.session_state:
    st.session_state.user_question = ""
    
text_input = st.text_input("Write your question", key="widget", on_change=submit)

if st.session_state.user_question and uploaded_file:
    with st.spinner(f"Processing question ..."):
        st.markdown(" ")
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        try:
            bot_reply = get_bot_reply(
                st.session_state["chat_history"][-3:],
                st.session_state.user_question,
                st.session_state["bot_engine"],
            )
            
            st.session_state["chat_history"].append(
                (st.session_state.user_question, bot_reply["answer"])
            )
            
        except RateLimitError:
            st.error(
                "Too many requests in a short period of time. Wait a minute an try again."
            )
        except InvalidRequestError:
            st.error(
                "The model maximum context length is 4097 tokens. However, the petition exceeded the limit."
            )
        except APIConnectionError:
            st.error("Error communicating with OpenAI. Connection aborted. Try again.")
            
        st.markdown(" ")
        
        for chat in st.session_state["chat_history"][::-1]:
            question = chat[0]
            answer = chat[1]
            question_tokens = get_num_tokens(question)
            answer_tokens = get_num_tokens(answer)
            
            st.info(question)
            st.caption(f"Tokens: {question_tokens}")
            
            if not any(
                word in question.lower().replace('"', "") for word in ["table", "tabla"]
            ):
                answer = answer.replace("\n", "\n\n")
                
            st.markdown(answer)
            st.caption(f"Tokens: {answer_tokens}")
            st.markdown(" ")
            st.markdown(" ")
