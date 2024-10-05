import json
import evaluate

def read_jsonl(file_path):
    """JSONL 파일에서 'output' 필드를 읽어들입니다."""
    outputs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if "output" in data:
                outputs.append(data["output"].strip())
    return outputs

def compare_outputs(file1, file2):
    """두 JSONL 파일에서 output을 비교하여 ROUGE-L 점수를 계산합니다."""
    outputs1 = read_jsonl(file1)
    outputs2 = read_jsonl(file2)

    rouge_scorer = evaluate.load('rouge')
    rouge_scores = rouge_scorer.compute(predictions=outputs2, references=outputs1)
    
    return rouge_scores


file1 = './results/instruct.json'
file2 = "./results/mpo.json"
rouge_l_scores = compare_outputs(file1, file2)

print(rouge_l_scores)
