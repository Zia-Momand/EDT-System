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
    """Combine the retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def chat_stream(response_text):
    """Simulate a chat stream output for the given response text."""
    for char in response_text:
        yield char
        time.sleep(0.005)

# --- Main Streamlit App ---

def main():
    st.markdown("##### Elderly Care Companion - Real-time Health Insights for your loved one")
    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # --- API keys ---
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    supabase_api_key = os.getenv("SUPABASE_API_KEY") or st.secrets.get("SUPABASE_API_KEY")
    supabase_url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")

    if not (openai_api_key and supabase_api_key and supabase_url):
        st.error("Please set OPENAI_API_KEY, SUPABASE_API_KEY, and SUPABASE_URL in your environment or Streamlit secrets.")
        return

    # --- LLM and Chains ---
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0)

    # 1. Standalone question prompt
    standalone_question_template = (
        "You are given conversation history and a current question related to elderly health sensor data "
        "(e.g., heart rate, blood oxygen (SpO₂), sleep stages, etc.). Your task is to convert this into a clear, standalone question.\n\n"
        "Instructions:\n"
        "- If the user asks for latest values, rephrase to ask for the most recent measurement.\n"
        "- If the user asks for trends, rephrase to ask for the average, min, max, and meaningful interpretation over time.\n"
        "- Be specific about the metric (heart rate, SpO₂, or sleep stage).\n\n"
        "Conversation history: {conv_history}\n"
        "Question: {question}\n"
        "Standalone question:"
    )
    standalone_prompt = PromptTemplate(
        template=standalone_question_template,
        input_variables=["conv_history", "question"]
    )

    # 2. Health Insight Answer Prompt
    answer_template = (
        "You are a helpful and caring assistant for caregivers, specialized in interpreting elderly health sensor data "
        "(heart rate, SpO₂, and sleep). Use the context to answer the caregiver's question simply and kindly.\n\n"
        "Instructions:\n"
        "- Summarize key metrics: latest, average, min, max.\n"
        "- Explain whether values are normal or need attention.\n"
        "- If helpful, suggest a visualization (e.g., 'A line chart of SpO₂ levels over the past 7 days would show the trend.')\n"
        "- Do not draw the chart, just describe it.\n"
        "- Be concise, empathetic, and avoid medical jargon.\n"
        "- If you cannot answer, say: 'I don’t have enough data to answer. Please contact help@elderlycare.com.'\n\n"
        "Context:\n{context}\n\n"
        "Conversation History:\n{conv_history}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
    answer_prompt = PromptTemplate(
        template=answer_template,
        input_variables=["context", "conv_history", "question"]
    )

    # 3. Weekly/Daily Summary Prompt
    summary_insight_template = (
        "You are summarizing sensor data (heart rate, SpO₂, sleep) for elderly health.\n"
        "Create a brief health summary for a caregiver based on the following data:\n\n"
        "{context}\n\n"
        "Summarize daily or weekly health trends, flag anomalies gently, and suggest one actionable step if appropriate.\n"
        "Avoid technical terms.\n\n"
        "Example format:\n"
        "- **Overall Summary**: [plain summary]\n"
        "- **Heart Rate**: [avg, any spikes]\n"
        "- **SpO₂**: [trend, any low values]\n"
        "- **Sleep**: [sleep quality, efficiency, disturbances]\n"
        "- **Care Tip**: [a practical suggestion]\n"
    )
    summary_prompt = PromptTemplate(
        template=summary_insight_template,
        input_variables=["context"]
    )

    # Chains
    standalone_chain = LLMChain(llm=llm, prompt=standalone_prompt)
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    # --- Vector store ---
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    supabase_client = create_client(supabase_url, supabase_api_key)
    vector_store = SupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name="documents",
        query_name="match_documents"
    )
    retriever = vector_store.as_retriever()

    # --- Chat Input ---
    # --- Pre-suggested summary options (like ChatGPT suggestions) ---
    if not st.session_state.messages:
        st.markdown("##### 💡 Quick Start")
       
        if st.button("📝 Weekly Health Summary"):
                with st.spinner("Generating caregiver summary..."):
                    docs = retriever.get_relevant_documents("weekly summary", k=6)
                    context = combine_documents(docs)
                    summary = summary_chain.run(context=context)
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.chat_message("assistant").markdown(summary)
       
        st.markdown(
                """
                Click this to get an AI-generated overview of the elder's sleep, heart rate, and SpO₂ condition over the past 7 days.
                """
            )

    user_input = st.chat_input("What is your question?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        conv_history = format_conv_history(st.session_state.messages)

        with st.spinner("Processing your question..."):
            standalone_question = standalone_chain.run(conv_history=conv_history, question=user_input)
            docs = retriever.get_relevant_documents(standalone_question, k=4)
            context = combine_documents(docs)
            answer = answer_chain.run(context=context, conv_history=conv_history, question=user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            streamed_text = ""
            for char in chat_stream(answer):
                streamed_text += char
                message_placeholder.markdown(streamed_text)
        st.session_state.messages.append({"role": "assistant", "content": streamed_text})
   

if __name__ == "__main__":
    main()
