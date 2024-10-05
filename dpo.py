import os
import torch

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset, load_dataset
from trl import DPOTrainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
# from utils import print_trainable_parameters
from collections import defaultdict
import json
import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='models/mpo')
    parser.add_argument('--paired_file', type=str, default="pair_for_mpo.json")
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.5)

    return parser

def main(args):
    output_dir = args.output_dir
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    paired_file = args.paired_file
    epochs = args.epochs
    beta = args.beta
    learning_rate = 1e-4

    train_data_dict = defaultdict(list)

    with open(paired_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            train_data_dict['prompt'].append(data['question'])
            train_data_dict['chosen'].append(data['chosen'].strip())
            train_data_dict['rejected'].append(data['rejected'].strip())
    dataset = Dataset.from_dict(dict(train_data_dict))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def return_prompt_and_responses(samples):
        return {
            "prompt": samples["prompt"],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    original_columns = dataset.column_names

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=epochs, 
        save_steps= 100,
        learning_rate=learning_rate,
        bf16=True,
        save_total_limit=3,
        logging_steps=10,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    # print_trainable_parameters(model)

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=2048,
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    main(args)