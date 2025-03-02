from dotenv import load_dotenv
import streamlit as st
st.set_page_config(layout="centered", initial_sidebar_state="expanded")
import os
import time
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.supabase import SupabaseVectorStore
from supabase import create_client

# Load environment variables from the .env file
load_dotenv()

# --- Helper functions ---

supabase_url = os.getenv("SUPABASE_URL")

def format_conv_history(messages):
    """Format conversation history as a simple string."""
    history = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        history += f"{role.capitalize()}: {content}\n"
    return history.strip()

def combine_documents(docs):
    """Combine the retrieved documents into a single context string.
    
    Assumes each document has a 'page_content' attribute.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def chat_stream(response_text):
    """
    Simulate a chat stream output for the given response text.
    Yields one character at a time with a fast delay.
    """
    for char in response_text:
        yield char
        time.sleep(0.005)  # Adjust delay for a fast simulation

# --- Main Streamlit App ---

def main():
    st.title("Ask about Elderly Loved One Health Status")

    # Initialize chat history in session_state if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # --- Set up API keys and clients ---
    # Retrieve keys from environment variables or st.secrets
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    supabase_api_key = os.getenv("SUPABASE_API_KEY") or st.secrets.get("SUPABASE_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")

    if not (openai_api_key and supabase_api_key and supabase_url):
        st.error("Please set OPENAI_API_KEY, SUPABASE_API_KEY, and SUPABASE_URL in your environment or Streamlit secrets.")
        return

    # Create the LLM instance (using ChatOpenAI)
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

    # --- Define prompt templates ---

    # 1. Standalone question prompt
    standalone_question_template = (
        "given a sensor reading calories data, you have read the data from vector embedding and Given some conversation history (if any) and a question, convert the question to a standalone question. \n"
        "conversation history: {conv_history}\n"
        "question: {question}\n"
        "standalone question:"
    )
    standalone_prompt = PromptTemplate(
        template=standalone_question_template,
        input_variables=["conv_history", "question"]
    )

    # 2. Answer generation prompt
    answer_template = (
        "You are a helpful and enthusiastic support bot who can answer a given question about Elderly care based on the context provided and conversation history. \n"
        "Try to find the answer in the context. If the answer is not given in the context, find answer in the conversation history if possible. \n"
        "If you really don't know the answer, say \"I'm sorry, I don't know the answer to that.\" And direct the questioner to email help@elderlycare.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.\n"
        "context: {context}\n"
        "conversation history: {conv_history}\n"
        "question: {question}\n"
        "answer:"
    )
    answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["context", "conv_history", "question"]
    )

    # Build the LLMChains for each step
    standalone_chain = LLMChain(llm=llm, prompt=standalone_prompt)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)

    # --- Set up Supabase-backed vector store as retriever ---
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    supabase_client = create_client(supabase_url, supabase_api_key)
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents"
    )
    retriever = vector_store.as_retriever()

    # --- Accept user input ---
    user_input = st.chat_input("What is your question?")
    if user_input:
        # Append the user’s question to the conversation history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Format conversation history for use in prompts
        conv_history = format_conv_history(st.session_state.messages)

        with st.spinner("Processing your question..."):
            # Step 1. Generate a standalone question using conversation history and the current question.
            standalone_question = standalone_chain.run(conv_history=conv_history, question=user_input)

            # Step 2. Retrieve relevant documents based on the standalone question.
            docs = retriever.get_relevant_documents(standalone_question, k=4)
            context = combine_documents(docs)

            # Step 3. Generate the final answer using the answer prompt.
            answer = answer_chain.run(context=context, conv_history=conv_history, question=user_input)

        # Display the assistant's answer as a streaming message.
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            streamed_text = ""
            for char in chat_stream(answer):
                streamed_text += char
                message_placeholder.markdown(streamed_text)
        st.session_state.messages.append({"role": "assistant", "content": streamed_text})

if __name__ == "__main__":
    main()
