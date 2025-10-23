import streamlit as st
from inference import get_integrated_rag_chain
from research_paper_loader import create_or_load_vector_store
from utils import CustomPrinter, get_device

# The @st.cache_resource decorator tells Streamlit to run this function only once,
# when the app first starts. It then caches the returned objects (our models and chain)
# in memory, so they don't have to be reloaded on every user interaction.
@st.cache_resource
def load_resources(base_model_id:str, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False, _c_print = print):
    # Create vector store before loading models
    vectorstore = create_or_load_vector_store(device=device, print_logs=print_logs, c_print=c_print)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    full_chain = get_integrated_rag_chain(
        model_id=base_model_id,
        model_save_path=finetuned_model_path,
        retriever=retriever,
        device=device,
        print_logs=print_logs,
        c_print=_c_print
    )
    st.success("Models and application are ready!")
    return full_chain

if __name__ == "__main__":
    BASE_MODEL_ID = "google/gemma-2b-it"
    MODEL_SAVE_PATH = ".model_training_cache/gemma-2b-it-summarizer/checkpoint-225"
    print_logs = True

    c_print = CustomPrinter()
    c_print.set_print_fc(st.write)
    
    device = get_device()
    c_print(f"Using device: {device}")

    # Streamlit UI configuration
    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("🤖 AI Research Assistant")
    st.info("Ask a question about my research papers, or ask for a summary (e.g., 'Summarize the RehabFork system').")

    # Load all the resources (this will be cached)
    try:
        full_chain = load_resources(
            base_model_id=BASE_MODEL_ID,
            finetuned_model_path=MODEL_SAVE_PATH,
            device=device,
            print_logs=print_logs,
            _c_print = c_print
        )
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