import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from inference import get_integrated_rag_chain
from utils import CustomPrinter, get_device

# The @st.cache_resource decorator tells Streamlit to run this function only once,
# when the app first starts. It then caches the returned objects (our models and chain)
# in memory, so they don't have to be reloaded on every user interaction.
@st.cache_resource
def load_resources(base_model_id:str, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False):
    full_chain = get_integrated_rag_chain(
        base_model_id=base_model_id,
        finetuned_model_path=finetuned_model_path,
        device=device,
        print_logs=print_logs
    )
    st.success("Models and application are ready!")
    return full_chain

if __name__ == "__main__":
    c_print = CustomPrinter()
    c_print.set_print_fc(st.write)
    
    # Streamlit UI configuration
    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("🤖 AI Research Assistant")
    st.info("Ask a question about my research papers, or ask for a summary (e.g., 'Summarize the RehabFork system').")

    # Load all the resources (this will be cached)
    try:
        full_chain = load_resources()
    except Exception as e:
        st.error(f"An error occurred during model loading: {e}")
        st.stop()

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            # Use st.write_stream for a beautiful "typing" effect
            response = st.write_stream(full_chain.stream(prompt))
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})