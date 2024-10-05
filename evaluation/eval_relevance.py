import json

file_path = "./results/mpo.json"

outputs = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if "output" in data:
            outputs.append(data["output"].strip())

relevance = 0
for output in outputs:
    if "Explanation:" in output:
        relevance+=1

print(relevance/817*100)