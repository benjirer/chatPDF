import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import openai
import base64
from PIL import Image
from copy import deepcopy

# page configuration
favicon = Image.open('pdf_icon.png')

st.set_page_config(
    page_title="chatPDF", 
    page_icon=favicon,
    initial_sidebar_state='collapsed')

error_box = st.empty()
# ------------------------------- INITIALIZE -------------------------------

# openai connection

# initialize session states 

# for chatbot
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = None

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'input' not in st.session_state:
    st.session_state.input = ''

# ------------------------------- HEADER -------------------------------
# title and logo

st.markdown(
    f'<div style="text-align: center"><img src="data:image/png;base64,{base64.b64encode(open("pdf_icon.png", "rb").read()).decode()}" width="50"><h1 style="text-align: center; color: black;">chat<span style="color: #6495ED">PDF</span></h1></div>', 
    unsafe_allow_html=True
)

adapt_streamlit_style = """
            <style>
            #MainMenu {visibility:hidden;}
            .reportview-container .sidebar-content {
                padding-top: 0rem;
            }
            .css-62i85d {
                visibility:hidden;
            }
            .css-15zrgzn {
                display:none
            }
            .css-eczf16 {
                display:none
            }
            .css-jn99sy {
                display:none
            }
            .block-container {
                padding-top: 3rem;
            }
            footer:{
                visibility:hidden;
            }
            footer:after{
                content:'by Benjirer';
                position:relative;
                padding:5px;
            }
            </style>
            """
st.markdown(adapt_streamlit_style, unsafe_allow_html=True) 

st.markdown("""---""")

st.markdown(
    f'<div style="text-align: center">Upload the PDF you would like to query.</div>', 
    unsafe_allow_html=True
)    
pdf_file = st.file_uploader("",type=["pdf"])


# ------------------------------- SIDEBAR -------------------------------

# Sidebar contents
with st.sidebar:
    # logo
    st.markdown(
        f'<div style="text-align: center"><img src="data:image/png;base64,{base64.b64encode(open("pdf_icon.png", "rb").read()).decode()}" width="50"><h1 style="text-align: center; color: black;">chat<span style="color: #6495ED">PDF</span></h1></div>', 
        unsafe_allow_html=True
    )
    st.write(" ")

    learn_more = st.expander("Learn More")
    with learn_more:
        st.write("**What is chatPDF?**")
        st.write("chatPDF lets you upload a pdf and query it via chatGPT.")

# ------------------------------- SCRIPT -------------------------------
if pdf_file:
    def new_chat():
        """
        Clears session state and starts a new chat.
        """
        # chatbot
        st.session_state["generated"] = []
        st.session_state["past"] = []
        st.session_state["input"] = ""
        st.session_state['prompt'] = None
            
    # new chat button

    st.markdown("""---""")
    col1, col2, col3 = st.columns([0.05,0.9,0.05])
    with col2:
        st.button("New Chat", on_click = new_chat, use_container_width=True)

    # ------------------------------- CHATBOT LAYOUT -------------------------------

    # layout of input/response containers
    response_container = st.container()
    input_container = st.container()
    warning_box = st.empty()
    warning_button = st.empty()

    # ------------------------------- CHATBOT FUNCTIONS -------------------------------

    # input
    def submit():
        st.session_state.input = st.session_state.widget
        st.session_state.widget = ''

    def get_text():
        input_text = st.chat_input("Ask something")
        return input_text

    os.environ["OPENAI_API_KEY"] = ""  # Set your OpenAI API key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())

    loader = PyMuPDFLoader(tmp_file.name)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    index = FAISS.from_documents(texts, embeddings)
    retriever = index.as_retriever()

    llm = ChatOpenAI(model_name='gpt-4', temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    summary = qa("Summarize the content to two sentences.")


    with response_container:
        with st.chat_message("assistant", avatar="pdf_icon.png"):
            st.markdown(summary["result"] + " How can I help you with that?")

    # applying the user input box
    with input_container:
        prompt = st.chat_input("Ask something")
        if st.session_state['prompt'] is None:
            st.session_state['prompt'] = prompt

    # chat display
    with response_container:

        for item in st.session_state['generated']:
            if item['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(item['content'], unsafe_allow_html=True) 
            else:
                with st.chat_message("assistant", avatar="pdf_icon.png"):
                    st.markdown(item['content'], unsafe_allow_html=True)

        if st.session_state['prompt'] is not None:

            with st.chat_message("user"):
                prompt_placeholder = st.empty()
                prompt_placeholder.markdown(st.session_state['prompt'], unsafe_allow_html=True)

            prompt = st.session_state['prompt']
            st.session_state['prompt'] = None
            
            # Display user question in chat message container
            prompt_placeholder.markdown("▌", unsafe_allow_html=True)

            prompt_placeholder.markdown(prompt)
            prompt_copy = deepcopy(prompt)

            prompt_placeholder.markdown(prompt, unsafe_allow_html=True)

            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar="pdf_icon.png"):
                message_placeholder = st.empty()
                message_placeholder.markdown("▌", unsafe_allow_html=True)
                full_response = ""
            
            
            full_response = qa(prompt)
            response = full_response['result']
            
            message_placeholder.markdown(response, unsafe_allow_html=True)

            st.session_state.generated.append({"role": "user", "content": prompt})
            st.session_state.generated.append({"role": "assistant", "content": response})

