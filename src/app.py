import streamlit as st
from models import get_vectorstore_from_url_or_pdfs, get_response
from langchain.schema import AIMessage, HumanMessage
# Streamlit app configuration
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar for user input
with st.sidebar:
    st.header("Settings")
    url_input = st.text_input("Website URL")
    pdf_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Submit"):
        if url_input:
            st.session_state.vector_store = get_vectorstore_from_url_or_pdfs(url_input)
        elif pdf_files:
            st.session_state.vector_store = get_vectorstore_from_url_or_pdfs(None, files=pdf_files)
        else:
            st.info("Please enter a website URL or upload PDF files.")

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# User input and response handling
user_query = st.chat_input("Type your message here...")
if user_query:
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Display conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
