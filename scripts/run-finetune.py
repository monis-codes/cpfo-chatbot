# scripts/run_finetune.py

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

# 1. Model and Tokenizer
base_model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Quantization Config (to run on smaller GPUs)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 3. LoRA Config
lora_config = LoraConfig(
    r=16,                  # Rank
    lora_alpha=32,         # Alpha
    lora_dropout=0.1,      # Dropout
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # Target modules for Mistral
)

# 4. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map="auto" # Automatically select GPU
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 5. Load Datasets
dataset = load_dataset('json', data_files={'train': 'data/training/train_dataset.json',
                                           'test': 'data/training/val_dataset.json'})

# 6. Set Training Arguments
training_args = TrainingArguments(
    output_dir="./models/results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

# 7. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=lora_config,
    dataset_text_field="instruction", # Assuming your JSON has "instruction" field
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# 8. Start Training
print("Starting fine-tuning...")
trainer.train()

# 9. Save the LoRA adapter
adapter_path = "./models/epfo-mistral-lora"
trainer.model.save_pretrained(adapter_path)
print(f"LoRA adapter saved to {adapter_path}")