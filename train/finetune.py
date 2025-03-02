import torch
import torch.distributed as dist
import os
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from qwen_vl.modeling_qwen_vl import QWenVLForCausalLM
from datasets import load_dataset

def load_training_dataset():
    try:
        dataset = load_dataset("simplescaling/s1K-1.1", split="train")
        return dataset
    except Exception as e:
        raise RuntimeError("Error loading dataset") from e

def load_model():
    model_name = "Qwen/Qwen2.5-72B-VL"
    try:
        model = QWenVLForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        for name, param in model.named_parameters():
            if "visual_encoder" in name:
                param.requires_grad = False
        return model
    except Exception as e:
        raise RuntimeError("Error loading model") from e

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        try:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
        except Exception as e:
            raise RuntimeError("Error setting up distributed training") from e

def apply_lora(model):
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "cross_modal_connector"],
            lora_dropout=0.1,
            bias="none"
        )
        return get_peft_model(model, lora_config)
    except Exception as e:
        raise RuntimeError("Error applying LoRA") from e

def train_loop(model, train_dataset):
    is_primary = not dist.is_initialized() or dist.get_rank() == 0
    training_args = TrainingArguments(
        output_dir="./models/qwen_vl_finetuned",
        per_device_train_batch_size=128,
        gradient_accumulation_steps=1,
        save_strategy="steps",
        save_steps=500,
        logging_steps=100,
        learning_rate=7e-6,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=True,
        optim="adamw_bnb_8bit",
        num_train_epochs=10,
        report_to="none",
        ddp_find_unused_parameters=False if dist.is_initialized() else None,
        save_total_limit=3
    )
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        trainer.train()
    except Exception as e:
        raise RuntimeError("Error during training") from e
    
    if dist.is_initialized():
        dist.barrier()
    
    if is_primary:
        try:
            torch.save(model.state_dict(), "./models/qwen_vl_finetuned/final_model.pt")
        except Exception as e:
            raise RuntimeError("Error saving model") from e

if __name__ == "__main__":
    setup_distributed()
    dataset = load_training_dataset()
    model = load_model()
    model = apply_lora(model)
    train_loop(model, dataset)
    if dist.is_initialized():
        dist.destroy_process_group()
