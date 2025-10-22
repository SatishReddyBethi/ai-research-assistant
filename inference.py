import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from langchain_huggingface import HuggingFacePipeline
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from research_paper_loader import c_print, load_llm, format_docs, create_or_load_vector_store, build_q_and_a_rag_chain

def get_q_and_a_rag_chain(model_id:str, device:str = "cpu", print_logs:bool = False):
    """
    Get the Retrieval-Augmented Generation (RAG) chain for Q&A using the specified LLM.
    Args:
        model_id (str): The identifier of the base LLM model.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        print_logs (bool): Whether to print logs during the process.
    Returns:
        rag_chain: The RAG chain for Q&A.
        base_model: The loaded base LLM model.
        tokenizer: The tokenizer associated with the loaded base LLM model.
    """
    base_llm, base_model, tokenizer = load_llm(model_id=model_id, device=device, max_new_tokens=512, print_logs=print_logs)
    rag_chain = build_q_and_a_rag_chain(base_llm, print_logs=print_logs)
    return rag_chain, base_model, tokenizer 

def load_finetuned_llm(base_model, tokenizer, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False):
    """
    Load the fine-tuned LLM by merging the PEFT-trained adapter weights into the base model.
    Args:
        base_model: The base LLM model.
        tokenizer: The tokenizer associated with the base LLM model.
        finetuned_model_path (str): Path to the fine-tuned model adapter weights.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
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

def get_finetuned_rag_chain(base_model, tokenizer, finetuned_model_path:str, device:str = "cpu", print_logs:bool = False):
    """
    Get the RAG chain that uses the fine-tuned LLM for summarization.
    Args:
        base_model: The base LLM model.
        tokenizer: The tokenizer associated with the base LLM model.
        finetuned_model_path (str): Path to the fine-tuned model adapter weights.
        device (str): The device to load the model onto (e.g., "cpu", "cuda", "xpu").
        print_logs (bool): Whether to print logs during the process.
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

if __name__ == "__main__":
    BASE_MODEL_ID = "google/gemma-2b-it"
    MODEL_SAVE_PATH = ".model_training_cache/gemma-2b-it-summarizer/checkpoint-225"
    print_logs = True
    print_sources = False

    # Determine the target device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        # Check if Intel XPU (GPU) is available
        device = "xpu"
    else:
        device =  "cpu"
    
    c_print(f"Using device: {device}")

    # Create vector store before loading models
    vectorstore = create_or_load_vector_store(device=device, print_logs=print_logs)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    rag_chain, base_model, tokenizer = get_q_and_a_rag_chain(model_id=BASE_MODEL_ID, device=device, print_logs=print_logs)
    summarizer_chain = get_finetuned_rag_chain(base_model, tokenizer, finetuned_model_path=MODEL_SAVE_PATH, device=device)

    # First, define a lambda function that checks the input query for keywords.
    is_summary_request = RunnableLambda(
        lambda x: "summarize" in x.lower() or "summary" in x.lower()
    )

    # Now, create the branch. It takes the input string, checks the condition,
    # and runs the appropriate chain with that same string as input.
    full_chain = RunnableBranch(
        (is_summary_request, summarizer_chain),
        rag_chain, # This is the fallback if the condition is false
    )

    # --- 7. Test the Integrated Application ---
    print("\n--- Testing the Router with a Q&A Query ---")
    qa_query = "What is the purpose of the RehabFork system?"
    print(f"Query: {qa_query}")
    print("Response (Streaming):")
    for chunk in full_chain.stream(qa_query):
        print(chunk, end="", flush=True)

    print("\n\n--- Testing the Router with a Summarization Query ---")
    summary_query = "Summarize the section on the RehabFork system."
    print(f"Query: {summary_query}")
    print("Response (Streaming):")
    for chunk in full_chain.stream(summary_query):
        print(chunk, end="", flush=True)

    print("\n\n--- Integration Complete ---")