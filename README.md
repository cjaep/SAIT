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
For PLM generation, set MODEL="plm"\\
For Instruct-tuned model generation, set MODEL="inst"\\
For Alignment-tuned model generation, set MODEL={PATH_TO_MODEL}
```sh
bash scripts/generation.sh
```
