from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, LoraConfig
import json
import argparse

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default='output.json')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model', type=str, default="plm")
    parser.add_argument('--method', type=str, default="mpo")

    return parser

def batch_writer(output_file, chosen, outputs, question):
    for i in range(len(question)):
        output_dict_example = {
            "chosen" : chosen[i],
            "rejected" : outputs[i],
            "question" : question[i],         
        }
        with open(f"{output_file}", "a") as _jsonl_file:
            _jsonl_file.write(json.dumps(output_dict_example))
            _jsonl_file.write("\n")
    return


def main(args):
    dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice")
    dataloader = DataLoader(dataset['validation'], batch_size=1, shuffle=False)

    if args.model=="plm":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B").to('cuda')
    elif args.model=="inst":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to('cuda')
    else:
        print("mpo model")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.model,
        ).to("cuda")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # model.generation_config.temperature=None
    model.generation_config.top_p=None

    for i, input_text in enumerate(tqdm(dataloader, desc="Predicting...")):
        question = input_text['question']
        outputs_chosen = input_text['mc1_targets']['choices'][0]
        input = tokenizer(question, max_length=1024, padding=True, return_tensors='pt').to("cuda")

        output_token_rejected = model.generate(**input, max_new_tokens=32, do_sample=True, temperature=5.0, pad_token_id=tokenizer.eos_token_id)
        outputs_rejected = tokenizer.batch_decode(output_token_rejected, skip_special_tokens=True)

        for k in range(len(outputs_rejected)):
            outputs_rejected[k] = outputs_rejected[k][len(question[0]):].strip()

        if args.method=="mpo":
            output_token_chosen = model.generate(**input, max_new_tokens=32, do_sample=False, num_beams=5, pad_token_id=tokenizer.eos_token_id)
            outputs_chosen = tokenizer.batch_decode(output_token_chosen, skip_special_tokens=True)   

            for k in range(len(outputs_chosen)):
                outputs_chosen[k] = outputs_chosen[k][len(question[0]):].strip()


        output_dict_example = {
            "output_file" : args.output_file,
            "chosen" : outputs_chosen,
            "outputs" : outputs_rejected,
            "question" : question,            
        }
        batch_writer(**output_dict_example)

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    
    main(args)