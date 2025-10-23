import logging
# Set the root logger to only show WARNINGs and above
logging.basicConfig(level=logging.WARNING)

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline#, BitsAndBytesConfig
import torch
import intel_extension_for_pytorch as ipex
from operator import itemgetter
from utils import get_device, format_docs, CustomPrinter, run_hugging_face_auth

def load_research_papers(data_path: str, print_logs: bool = False, c_print = print):
    """
    Load and split research papers from the specified directory.
    Args:
        data_path (str): The path to the directory containing research papers in PDF format.
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        List[Document]: A list of Document objects representing the split chunks of the research papers.
    """
    all_docs_data = []

    if print_logs:
        c_print(f"Loading research papers from '{data_path}' folder...")

    for dirpath, dirnames, filenames in os.walk(data_path):
        for file in filenames:
            full_file_path = os.path.join(dirpath, file)
            loader = PyPDFLoader(full_file_path)
            docs = loader.load()
            # The.load() method returns a list of Document objects (one for each page).
            # We use.extend() to add all pages from the current PDF to our main list.
            all_docs_data.extend(docs)
            if print_logs:
                c_print(f"\n{len(docs)} Pages loaded from: {full_file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    if print_logs:
        c_print(f"\nSplitting {len(all_docs_data)} pages into chunks...")
    
    splits = text_splitter.split_documents(all_docs_data)
    
    if print_logs:
        c_print(f"Created {len(splits)} chunks.")
        # c_print(f"\nContents of a single chunk:\n{splits[0].page_content}")
        # c_print(f"\nMetadata of the chunk:\n{splits[0].metadata}")    
    return splits

def create_or_load_vector_store(model_name: str = "all-MiniLM-L6-v2", device: str = "cpu", persist_directory: str = ".chromaDB", print_logs: bool = False, verify_vector_store: bool = False, c_print = print):
    """
    Create or load the vector store from research papers.
    Args:
        model_name (str): The name of the embedding model to use. By default, we use 'all-MiniLM-L6-v2' as it is a popular and efficient model that runs locally.
        device (str): The device to run the embedding model on. Options are 'cpu', 'cuda', or 'xpu' (for Intel XPU).
        persist_directory (str): The directory where the vector store is persisted.
        print_logs (bool): Whether to print logs during the process.
        test_vector_store (bool): Whether to perform a test query on the vector store after loading/creation.
        c_print: CustomPrinter instance for logging.
    Returns:
        vectorstore (Chroma): The loaded or newly created vector store.
    """
    if print_logs:
        c_print("\nInitializing embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name = model_name,
        model_kwargs = {'device': device} # Use 'cuda' if you have a GPU
    )

    if os.path.exists(persist_directory):
        if print_logs:
            c_print(f"Loading existing vector store from '{persist_directory}' folder...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        if print_logs:
            c_print("Vector store loaded successfully.")
    else:
        splits = load_research_papers(data_path="./data", print_logs=print_logs, c_print=c_print)
        if print_logs:
            c_print("\nNo existing vector store found.")
            c_print(f"Creating new vector store in '{persist_directory}' folder...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        if print_logs:
            c_print("New vector store created and persisted.")
    
    if verify_vector_store:
        # Check if the vector store is working correctly
        c_print("\n--- Testing the vector store with a similarity search ---")
        test_query = "What is the main contribution of the 'Exergames for telerehabilitation' thesis?"
        retrieved_docs = vectorstore.similarity_search(test_query, k=2) # Retrieve the top 2 most relevant chunks

        c_print(f"Query: '{test_query}'")
        c_print(f"Retrieved {len(retrieved_docs)} documents.")

        for i, doc in enumerate(retrieved_docs):
            c_print(f"\n--- Document {i+1}: ---")
            c_print(doc.page_content)
            c_print("\n--- Metadata: ---")
            c_print(doc.metadata)
    
    return vectorstore

def load_model(model_id: str = "google/gemma-2b-it", device:str = "cpu", print_logs: bool = False, c_print = print):
    """
    Load the local LLM model and tokenizer from Hugging Face.
    Args:
        model_id (str): The Hugging Face model ID to load. By default, we use 'google/gemma-2b-it' as it's powerful and runs locally.
        device (str): The device to run the model on. Options are 'cpu', 'cuda', or 'xpu' (for Intel XPU).
        hf_env_var (str): The environment variable name that contains the Hugging Face API token.
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
    """
    run_hugging_face_auth()
    
    if print_logs:
        c_print(f"\nInitializing local LLM ({model_id})...")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # QLoRA Config for Memory Efficiency ---
    # NOTE: QLoRA on the XPU backend is cutting-edge and can sometimes be unstable.
    # If you encounter errors during the `trainer.train()` step, the first and best
    # troubleshooting step is to comment out this `bnb_config` and the
    # `quantization_config` argument in the model loading step below.
    # This will force training in bfloat16, which uses more memory but is more stable.
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16, # Same range as float32 but less precision, saves memory
        # quantization_config=bnb_config, # Using QLoRA for memory saving. Remove if training fails.
    )
    
    # Optimize the model for XPU inference or training
    # This should be done before the model is used
    if print_logs:
        c_print(f"Moving model to {device}...")
    
    if device != "cpu":
        model.to(device)
    
    if print_logs:
        c_print(f"Optimizing model for {device}...")
    
    if device == "xpu":
        # Only optimize if using Intel XPU
        model = ipex.optimize(model, dtype=torch.bfloat16) #, weights_prepack=False
    
    return model, tokenizer

def load_llm(model_id: str = "google/gemma-2b-it", device:str = "cpu", max_new_tokens:int = 512, print_logs: bool = False, c_print = print):
    """
    Load the local LLM using Hugging Face transformers and wrap it in a LangChain HuggingFacePipeline.
    Args:
        model_id (str): The Hugging Face model ID to load. By default, we use 'google/gemma-2b-it' as it's powerful and runs locally.
        device (str): The device to run the model on. Options are 'cpu', 'cuda', or 'xpu' (for Intel XPU).
        hf_env_var (str): The environment variable name that contains the Hugging Face API token.
        print_logs (bool): Whether to print logs during the process.
    Returns:
        llm (HuggingFacePipeline): The loaded local LLM wrapped in a LangChain HuggingFacePipeline.
    """
    model, tokenizer = load_model(
        model_id=model_id,
        device=device,
        print_logs=print_logs,
        c_print=c_print
    )    

    # Create a LangChain text-generation pipeline from the transformers library
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens, # The maximum number of tokens to generate
        temperature=0.1
    )

    # Wrap the transformers pipeline in a LangChain object
    llm = HuggingFacePipeline(pipeline=pipe)

    if print_logs:
        c_print("LLM initialized successfully.")

    return llm, model, tokenizer

def build_q_and_a_rag_chain(llm, print_logs: bool = False, c_print = print):
    """
    Build a Retrieval-Augmented Generation (RAG) chain for Q&A using the provided LLM.
    Args:
        llm (HuggingFacePipeline): The local LLM for generating answers.
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        rag_chain: The constructed RAG chain.
    """
    # This prompt is instructs the LLM to answer questions 'only' based on
    # the provided context from the papers. This is a key to prevent hallucinations.
    template = """
    You are an expert research assistant. Your task is to answer questions based on the provided context.
    Answer the question based only on the following context.
    If the answer cannot be found in the context, write "I am sorry! I could not find the answer in the provided documents.
    If you find multiple answers, combine them into a comprehensive response."

    Context:
    {context}

    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "question": itemgetter("question")
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def run_rag_pipeline(rag_chain, query, retriever, stream_output:bool = True, print_logs: bool = False, print_sources: bool = False, c_print = print):
    """
    Run the RAG pipeline with the given query.
    Args:
        rag_chain: The RAG chain to use for answering the query.
        query (str): The user's question to answer.
        retriever: The document retriever.
        stream_output (bool): Whether to stream the output token by token.
        print_logs (bool): Whether to print logs during the process.
        print_sources (bool): Whether to print the source documents used for answering.
        c_print: CustomPrinter instance for logging.
    Returns:
        None
    """
    if print_logs:
        c_print(f"Query: {query}\n")
        c_print("Retrieving relevant documents...")
    
    retrieved_docs = retriever.invoke(query)
    chain_input = {"context": retrieved_docs, "question": query}

    if stream_output:
        if print_logs:
            c_print("Answer (Streaming):")
        full_response = stream_rag_chain(rag_chain, chain_input, print_logs=print_logs, c_print=c_print)
    else:
        if print_logs:
            c_print("Answer:")        
        full_response = rag_chain.invoke({"question": query})
        if print_logs:
            c_print(full_response)

    if print_logs and print_sources:
        # This can be printed before answer generation (if needed)
        c_print("\nSources: ---")
        for i, doc in enumerate(retrieved_docs):
            c_print(f"Source {i+1} (from '{doc.metadata.get('source', 'N/A')}', page {doc.metadata.get('page', 'N/A')}):")
            c_print(f"\"{doc.page_content[:250]}...\"\n")

    return full_response


def stream_rag_chain(rag_chain, query:str, print_logs: bool = False, c_print = print):
    """
    Stream the output of the RAG chain for a given query.
    Args:
        rag_chain: The RAG chain to use for generation.
        query (str): The input query.
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        full_response (str): The complete response generated by the RAG chain.
    """
    full_response = ""
    for chunk in rag_chain.stream(query):
        if print_logs:
            # Print each chunk as it arrives
            if c_print == print:
                print(chunk, end="", flush=True)
            else:
                c_print(chunk, end="", flush=True, no_prefix=True)
        full_response += chunk

    # Finish the stream and go to next line
    if c_print == print:
        print("")
    else:
        c_print("", no_prefix=True)
    return full_response

if __name__ == "__main__":
    print_logs = True
    print_sources = False
    c_print = CustomPrinter()
    device = get_device()
    print(f"Using device: {device}")
    vectorstore = create_or_load_vector_store(device=device, print_logs=print_logs, c_print=c_print)
    llm, _, _ = load_llm(device=device, max_new_tokens=512, print_logs=print_logs, c_print=c_print)
    rag_chain = build_q_and_a_rag_chain(llm, print_logs=print_logs, c_print=c_print)
    # Check if the RAG chain is working as expected
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    query = "What is the purpose of the RehabFork system described in the papers?"
    response = run_rag_pipeline(rag_chain, query, retriever, print_logs=print_logs, print_sources=print_sources, c_print=c_print)
    if not print_logs:
        print(f"Q: {query}\nA: {response}")
