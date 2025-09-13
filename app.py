import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Page settings
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot - Powered by LangChain & OpenAI")

# Sidebar for API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to continue.")
else:
    # Initialize model
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")

    # Memory to store conversation
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    # Chat UI
    user_input = st.chat_input("Type your message here...")
    if user_input:
        response = conversation.predict(input=user_input)

        # Store chat history
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(response)

    # Display conversation
    for msg in st.session_state.memory.chat_memory.messages:
        if msg.type == "human":
            st.chat_message("user").markdown(msg.content)
        else:
            st.chat_message("assistant").markdown(msg.content)
