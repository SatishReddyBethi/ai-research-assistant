import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from research_paper_loader import load_llm, create_or_load_vector_store, build_q_and_a_rag_chain, stream_rag_chain
from utils import CustomPrinter, get_device, format_docs

def get_q_and_a_rag_chain(model_id:str, device:str = "cpu", print_logs:bool = False, c_print = print):
    """
    Get the Retrieval-Augmented Generation (RAG) chain for Q&A using the specified LLM.
    Args:
        model_id (str): The identifier of the base LLM model.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        rag_chain: The RAG chain for Q&A.
        base_model: The loaded base LLM model.
        tokenizer: The tokenizer associated with the loaded base LLM model.
    """
    base_llm, base_model, tokenizer = load_llm(model_id=model_id, device=device, max_new_tokens=512, print_logs=print_logs)
    rag_chain = build_q_and_a_rag_chain(base_llm, print_logs=print_logs)
    return rag_chain, base_model, tokenizer 

def load_finetuned_llm(base_model, tokenizer, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False, c_print = print):
    """
    Load the fine-tuned LLM by merging the PEFT-trained adapter weights into the base model.
    Args:
        base_model: The base LLM model.
        tokenizer: The tokenizer associated with the base LLM model.
        finetuned_model_path (str): Path to the fine-tuned model adapter weights.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        c_print: CustomPrinter instance for logging.
    Returns:
        finetuned_llm: The fine-tuned LLM wrapped in a HuggingFacePipeline.
    """
    if print_logs:
        c_print(f"Loading fine-tuned adapter from '{finetuned_model_path}'...")
    
    finetuned_model = PeftModel.from_pretrained(base_model, finetuned_model_path)
    
    if print_logs:
        c_print("Fine-tuned model loaded successfully.")

    # One pipeline for the fine-tuned model (for summarization)
    finetuned_llm_pipe = pipeline(
        "text-generation",
        model=finetuned_model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=256, # Summaries are shorter
        temperature=0.1
    )
    finetuned_llm = HuggingFacePipeline(pipeline=finetuned_llm_pipe)
    return finetuned_llm

def get_finetuned_rag_chain(base_model, tokenizer, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False, c_print = print):
    """
    Get the RAG chain that uses the fine-tuned LLM for summarization.
    Args:
        base_model: The base LLM model.
        tokenizer: The tokenizer associated with the base LLM model.
        finetuned_model_path (str): Path to the fine-tuned model adapter weights.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        summarizer_chain: The RAG chain for summarization using the fine-tuned LLM.
    """
    finetuned_llm = load_finetuned_llm(base_model, tokenizer, finetuned_model_path, device=device, print_logs=print_logs)
    # Chain 2: Summarizer (uses the FINE-TUNED model)
    summarizer_prompt_template = "### Input:\n{context}\n\n### Output:\n"
    summarizer_prompt = ChatPromptTemplate.from_template(summarizer_prompt_template)
    summarizer_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | summarizer_prompt
    | finetuned_llm
    | StrOutputParser()
    )

    return summarizer_chain

def get_integrated_rag_chain(model_id:str, model_save_path:str, device:str = "cpu", print_logs:bool = False, c_print = print):
    """
    Get the integrated RAG chain that routes between Q&A and Summarization based on the input query.
    Args:
        model_id (str): The identifier of the base LLM model.
        model_save_path (str): Path to the fine-tuned model adapter weights.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        print_logs (bool): Whether to print logs during the process.
        c_print: CustomPrinter instance for logging.
    Returns:
        full_chain: The integrated RAG chain with routing.
    """
    rag_chain, base_model, tokenizer = get_q_and_a_rag_chain(model_id=model_id, device=device, print_logs=print_logs)
    summarizer_chain = get_finetuned_rag_chain(base_model, tokenizer, finetuned_model_path=model_save_path, device=device)

    # Define a lambda function that checks the input query for keywords.
    is_summary_request = RunnableLambda(
        lambda x: "summarize" in x.lower() or "summary" in x.lower()
    )

    # Create the RunnableBranch that routes based on the condition.
    full_chain = RunnableBranch(
        (is_summary_request, summarizer_chain),
        rag_chain, # This is the fallback if the condition is false
    )
    return full_chain

if __name__ == "__main__":
    BASE_MODEL_ID = "google/gemma-2b-it"
    MODEL_SAVE_PATH = ".model_training_cache/gemma-2b-it-summarizer/checkpoint-225"
    print_logs = True
    print_sources = False

    c_print = CustomPrinter()
    device = get_device()    
    c_print(f"Using device: {device}")

    # Create vector store before loading models
    vectorstore = create_or_load_vector_store(device=device, print_logs=print_logs)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Build the Integrated RAG Chain
    full_chain = get_integrated_rag_chain(
        model_id=BASE_MODEL_ID,
        model_save_path=MODEL_SAVE_PATH,
        device=device,
        print_logs=print_logs
    )

    # Test the Integrated Application
    c_print("\nTesting the Router with a Q&A Query")
    qa_query = "What is the purpose of the RehabFork system?"
    c_print(f"Query: {qa_query}")
    c_print("Response (Streaming):")
    full_response = stream_rag_chain(full_chain, qa_query)

    c_print("\n\nTesting the Router with a Summarization Query")
    summary_query = "Summarize the section on the RehabFork system."
    c_print(f"Query: {summary_query}")
    c_print("Response (Streaming):")
    full_response = stream_rag_chain(full_chain, summary_query)
