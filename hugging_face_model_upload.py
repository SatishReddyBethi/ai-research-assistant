from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import run_hugging_face_auth
import os

if __name__ == "__main__":
    BASE_MODEL_ID = "google/gemma-2b-it"
    FINETUNED_MODEL_PATH = ".model_training_cache/gemma-2b-it-summarizer/checkpoint-225"

    run_hugging_face_auth()
    
    hf_username = os.getenv("HUGGINGFACE_USERNAME")
    NEW_MODEL_NAME = f"{hf_username}/gemma-2b-it-summarizer-research-assistant"

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16,
    )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)

    print("Pushing to Hugging Face Hub...")
    # This will create a new repository on your Hugging Face profile
    model.push_to_hub(NEW_MODEL_NAME)

    # Also upload the tokenizer so it's bundled with the model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.push_to_hub(NEW_MODEL_NAME)

    print(f"Successfully uploaded model and tokenizer to: {NEW_MODEL_NAME}")