import streamlit as st
from inference import get_integrated_rag_chain
from q_and_a_rag_model import create_or_load_vector_store
from utils import get_device, get_env
import threading
import queue

# The @st.cache_resource decorator tells Streamlit to run this function only once,
# when the app first starts. It then caches the returned objects (our models and chain)
# in memory, so they don't have to be reloaded on every user interaction.
@st.cache_resource
def load_resources(base_model_id:str, finetuned_model_id:str = "", finetuned_model_path:str = "", print_logs:bool = False):
    device = get_device()
    print(f"Using device: {device}")
    
    use_xpu = get_env("USE_XPU")

    if device == "xpu" and not use_xpu:
        device = "cpu"
        print(f"USE_XPU is set to {use_xpu} so switching device to 'cpu'")

    if device == "xpu":
        # import ipex even if not used as it loads all the required optimization for XPU
        import intel_extension_for_pytorch as ipex

    # Create vector store before loading models
    vectorstore = create_or_load_vector_store(device=device, print_logs=print_logs)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    full_chain = get_integrated_rag_chain(
        model_id=base_model_id,
        finetuned_model_id=finetuned_model_id,
        finetuned_model_path=finetuned_model_path,
        retriever=retriever,
        device=device,
        print_logs=print_logs
    )
    return full_chain

@st.cache_resource
def init_inference(_rag_chain):
    # Run a test inference to initialize the model allocated memory (first run is always the slowest)
    # This will also act as a test to know if the model is working as expected
    response = _rag_chain.invoke("What is the purpose of the RehabFork system?")
    print(response)

# Worker function to run the LLM chain in a separate thread
def run_chain_in_thread(chain, query, q, stop_event):
    """
    Target function for the background thread.
    Streams the chain's response and puts tokens into a queue.
    """
    try:
        for chunk in chain.stream(query):
            if stop_event.is_set():
                # If the stop event is set, break the loop
                break
            q.put(chunk)
    except Exception as e:
        q.put(f"Error: {e}")
    finally:
        # Signal that the stream has finished
        q.put(None)

if __name__ == "__main__":
    hf_username = get_env("HUGGINGFACE_USERNAME")  
    FINETUNED_MODEL_ID = f"{hf_username}/gemma-2b-it-summarizer-research-assistant"
    # If a fine tuned model id is not given, check for local model
    if FINETUNED_MODEL_ID:
        FINETUNED_MODEL_PATH = ""
    else:
        FINETUNED_MODEL_PATH = get_env("LOCAL_FINETUNED_MODEL_PATH")
    BASE_MODEL_ID = "google/gemma-2b-it"
    print_logs = True

    # Initialize Streamlit UI
    # Set the page configuration as the first Streamlit command
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🤖",
        layout="wide"
    )

    # Hide default Streamlit elements for a cleaner look
    hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    descripton = """
    This app is an AI-powered chatbot that can answer questions about my research papers and provide expert-style summaries of technical sections.
    """
    with st.sidebar:
        st.header("🤖 AI Research Assistant")
        st.info(descripton)
        st.markdown("---")
        st.link_button("My GitHub", "https://github.com/SatishReddyBethi")
        st.link_button("My LinkedIn", "https://linkedin.com/in/bethi-satish-reddy")

    st.title("🤖 AI Research Assistant")
    st.subheader("- By Satish Bethi")
    # -----------------------------------
    st.divider()
    st.text(descripton)
    # -----------------------------------
    st.divider()
    st.info("Ask a question about my research papers, or ask for a summary or request a summary (e.g., 'What is the purpose of the RehabFork system?' or 'Summarize the section on RehabFork').")

    # Load all the resources (this will be cached)
    with st.spinner("Initializing the AI Assistant... This may take a moment."):
        try:
            full_chain = load_resources(
                base_model_id=BASE_MODEL_ID,
                finetuned_model_id=FINETUNED_MODEL_ID,
                finetuned_model_path=FINETUNED_MODEL_PATH,
                print_logs=print_logs
            )

            init_inference(
                _rag_chain=full_chain
            )
            st.success("🤖 AI Assistant is ready!")
        except Exception as e:
            st.error(f"An error occurred during initialization: {e}")
            st.stop()

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "generating" not in st.session_state:
        st.session_state.generating = False

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
    
    # This block handles the user input and STARTS the generation process
    if prompt := st.chat_input("What would you like to know?"):
        if not st.session_state.generating:
            # Add user message to history and display it
            st.session_state.messages.append({"role": "user", "content": prompt, "avatar": "🧑‍💻"})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
                print(f"User Prompt 🧑‍💻:\n{prompt}")

            # Start the generation process in the background
            st.session_state.generating = True
            st.session_state.stop_event = threading.Event()
            q = queue.Queue()
            st.session_state.queue = q
            
            st.session_state.thread = threading.Thread(
                target=run_chain_in_thread,
                args=(full_chain, prompt, q, st.session_state.stop_event)
            )
            st.session_state.thread.start()
            
            # Rerun the script to update the UI
            st.rerun()

    # This block handles displaying the streaming response and the stop button
    if st.session_state.generating:
        full_response = ""
        # Display the stop generating button
        if st.button("⏹️ Stop generating"):
            if st.session_state.stop_event:
                st.session_state.stop_event.set()
            st.session_state.generating = False
            print("\nUser stopped generating!")
            st.rerun()
    
        with st.status("Generating..."):
            response_container = st.empty() 
            print(f"Generated Answer 🤖:")
            # Continuously check the queue for new tokens
            while True:
                try:
                    token = st.session_state.queue.get(block=False)
                    if token is None:
                        st.session_state.generating = False
                        break
                    full_response += token
                    response_container.markdown(full_response + "▌")
                    print(token, end="", flush=True)
                except queue.Empty:
                    # If the queue is empty, check if the thread is still running
                    if not st.session_state.thread.is_alive():
                        st.session_state.generating = False
                        break
                    # Briefly pause to prevent a tight loop from consuming too much CPU
                    import time
                    time.sleep(0.1)
            
        # # Update status to complete
        # st.status.update(
        #     label="Generation Completed!", state="complete", expanded=False
        # )

        with st.chat_message("assistant", avatar="🤖"):
            # Display final response and add to history
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        
        # Rerun one last time to clear the stop button after generation finishes
        if not st.session_state.generating:
            st.rerun()