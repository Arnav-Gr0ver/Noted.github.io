import torch
import os
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl.modeling_qwen_vl import QWenVLForCausalLM
from datasets import load_dataset

def load_training_dataset():
    return load_dataset("simplescaling/s1K-1.1", split="train", num_proc=8)

def load_model():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = QWenVLForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-72B-VL",
        quantization_config=quantization_config,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    return prepare_model_for_kbit_training(model)

def apply_lora(model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "cross_modal_connector"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)

def train_loop(model, train_dataset):
    training_args = TrainingArguments(
        output_dir="./models/qwen_vl_finetuned",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_strategy="steps",
        save_steps=2000,
        logging_steps=500,
        learning_rate=7e-6,
        weight_decay=0.01,
        warmup_steps=500,
        bf16=True,
        optim="adamw_bnb_8bit",
        max_steps=50000,
        report_to="none",
    )
    Trainer(model=model, args=training_args, train_dataset=train_dataset).train()
    torch.save(model.state_dict(), "./models/qwen_vl_finetuned/final_model.pt")

if __name__ == "__main__":
    dataset = load_training_dataset()
    model = load_model()
    model = apply_lora(model)
    train_loop(model, dataset)
