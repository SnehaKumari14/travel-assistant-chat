import streamlit as st
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from rag_utils import load_documents, create_vector_db, get_conversational_rag_chain, get_chat_model

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Travel Assistant Chat",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = []

# Initialize vector database if not already done
if st.session_state.vector_db is None:
    with st.spinner("Loading documents and creating vector database..."):
        docs = load_documents()
        st.session_state.vector_db = create_vector_db(docs)
        st.success("Vector database created successfully!")

# Sidebar for controls
with st.sidebar:
    st.title("Travel Assistant")
    st.write("Your AI travel companion")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        clear_chat_history()
        st.rerun()

# Main chat interface
st.title("Travel Assistant Chat")

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Ask me anything about travel..."):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chat = get_chat_model()
            conversation_rag_chain = get_conversational_rag_chain(st.session_state.vector_db, chat)
            
            # Create a placeholder for streaming response
            response_placeholder = st.empty()
            response_message = ""
            
            # Stream the response
            for chunk in conversation_rag_chain.pick("answer").stream(
                {"messages": st.session_state.messages[:-1], "input": prompt}
            ):
                response_message += chunk
                response_placeholder.write(response_message)
            
            # Add the complete response to chat history
            st.session_state.messages.append(AIMessage(content=response_message)) 