from typing import List

import fire
import torch
import transformers
from datasets import load_dataset


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PrefixTuningConfig,
    TaskType
)
from transformers import AutoTokenizer, AutoModelForCausalLM


def train(
    # model/data params
    base_model: str = "meta-llama/Llama-3.2-1B", 
    output_dir: str = "models/sft",

    micro_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    val_set_size: int = 100,
    
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    ):

    device_map = "auto"


    # Step 1: Load the model and tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # load_in_8bit=True, # Add this for using int8
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    #   Add this for training LoRA

    config = LoraConfig(
          r=lora_r,
          lora_alpha=lora_alpha,
          target_modules=lora_target_modules,
          lora_dropout=lora_dropout,
          bias="none",
          task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # model = prepare_model_for_int8_training(model) # Add this for using int8

    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    

    def tokenize(data):
        source_ids = tokenizer.encode(data['question'])
        target_ids = tokenizer.encode(data['best_answer'])

        input_ids = source_ids + target_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    train_val = dataset["validation"].train_test_split(
        test_size=val_set_size, shuffle=False, seed=42
    )
    # print(train_val["train"]["question"])
    train_data = (
        train_val["train"].map(tokenize)
    )
    val_data = (
        train_val["test"].map(tokenize)
        
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)