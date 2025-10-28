import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from q_and_a_rag_model import load_model
from utils import CustomPrinter, get_device, get_env

def format_training_prompt(example):
    # This structured format is crucial for instruction-tuned models like gemma
    return {"text": f"### Input:\n{example['input']}\n\n### Output:\n{example['output']}"}

def fine_tune_model(model_id:str, dataset_file_path:str, finetuned_model_path:str , device:str = "cpu", print_logs:bool = False, c_print = print):
    model, tokenizer = load_model(
        model_id=model_id,
        device=device
    )

    # Set padding token to end-of-sequence token for open-ended generation
    tokenizer.pad_token = tokenizer.eos_token

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Load and Prepare the Dataset
    dataset = load_dataset("json", data_files=dataset_file_path, split="train")
    dataset = dataset.map(format_training_prompt)

    # SFT Configuration
    # This object holds all the training parameters and configurations.
    sft_config = SFTConfig(
        output_dir=finetuned_model_path,
        logging_dir=f"{finetuned_model_path}/logs",
        logging_steps=10,
        learning_rate=2e-4,
        save_strategy="epoch",
        
        dataset_text_field="text",
        packing=False,
        
        bf16=False, # Bypasses the faulty hardware check
        gradient_checkpointing=True, # Saves memory during training
        num_train_epochs=10,
        per_device_train_batch_size=1, # Lower this to 2 or 1 if you run out of VRAM
        max_length=256,
        gradient_accumulation_steps=8,
        # optim="paged_adamw_8bit", # Use 8-bit Adam optimizer for memory efficiency
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        processing_class=tokenizer
    )

    use_xpu = get_env("USE_XPU")

    if device == "xpu" and not use_xpu:
        device = "cpu"
        print(f"USE_XPU is set to {use_xpu} so switching device to 'cpu'")

    if device == "xpu":
        # import ipex even if not used as it loads all the required optimization for XPU
        import intel_extension_for_pytorch as ipex
        # Empty the XPU cache (if any) to free up memory before training
        torch.xpu.empty_cache()

    if print_logs:
        c_print("\nStarting Fine-Tuning")
    
    trainer.train()
    
    if print_logs:
        c_print("Fine-Tuning Complete")

    trainer.save_model(finetuned_model_path)
    
    if print_logs:
        c_print(f"Fine-tuned model adapter saved to '{finetuned_model_path}'")
    
    return trainer, tokenizer


if __name__ == "__main__":
    BASE_MODEL_ID = "google/gemma-2b-it"
    DATASET_FILE_PATH = "fine_tuning_dataset_2.jsonl"
    FINETUNED_MODEL_PATH = ".model_training_cache/gemma-2b-it-summarizer"
    print_logs = True
    print_sources = False
    c_print = CustomPrinter()
    device = get_device()
    c_print(f"Using device: {device}")

    # Fine-tune the model
    trainer, tokenizer = fine_tune_model(
        model_id=BASE_MODEL_ID,
        dataset_file_path=DATASET_FILE_PATH,
        finetuned_model_path=FINETUNED_MODEL_PATH,
        device=device,
        print_logs=print_logs,
        c_print=c_print
    )
    
    # Test the Fine-Tuned Model for Immediate Validation
    c_print("\nTesting the fine-tuned model")

    # Example paragraph for testing
    test_paragraph = """What are all the papers authored by Satish Reddy Bethi? Can you summarize each paper one by one?"""

    # Format the test prompt exactly as the model was trained
    prompt_text = f"### Input:\n{test_paragraph}\n\n### Output:\n"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Generate the summary using the fine-tuned model
    # Note: We don't need to load the model again, 'trainer.model' is the fine-tuned model.
    outputs = trainer.model.generate(**inputs, max_new_tokens=100)
    summary = tokenizer.decode(outputs, skip_special_tokens=True)

    c_print("Generated Summary: ---")
    # The output will include your input prompt, so we can print just the generated part.
    generated_text = summary.split("### Output:\n")[-1]

    c_print(f"Test Paragraph:\n{test_paragraph}\nSummary:{generated_text}")