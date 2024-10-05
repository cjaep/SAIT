# SAIT
Code for SAIT interview

## Installation 
Code is based on Huggingface's `transformers>=4.35.0`.
```bash
conda create -n sait python=3.8
conda activate sait
git clone https://github.com/cjaep/SAIT.git
cd SAIT
pip install -r requirements.txt
```

### 1. SFT
```sh
bash scripts/sft.sh
```

### 2. Pair Generation for Alignment-tuning
For MPO training, set method="mpo"<br/>
For DPO training, set method="dpo"
```sh
bash scripts/pair_generation.sh
```

### 3. Alignment-tuning
```sh
bash scripts/dpo.sh
```

### 4. Generation
For PLM generation, set MODEL="plm"<br/>
For Instruct-tuned model generation, set MODEL="inst"<br/>
For Alignment-tuned model generation, set MODEL={PATH_TO_MODEL}
```sh
bash scripts/generation.sh
```
