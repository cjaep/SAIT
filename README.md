# SAIT
Code for SAIT interview

## Installation 
Our code is based on Huggingface's `transformers>=4.35.0`.

```bash
conda create -n sait python=3.8
conda activate sait
git clone https://github.com/cjaep/SAIT.git
cd SAIT
pip install -r requirements.txt
```

### 2. Generation
MODEL="plm" -> PLM generation
MODEL="inst" -> Instruct-tuned model generation
MODEL={TRAINED_DPO_MODEL} -> Alignment-tuned model generation
```sh
bash scripts/generation.sh
```
