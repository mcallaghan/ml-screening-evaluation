#!/bin/bash

#SBATCH --job-name=ml-screening-pipeline-2-70b
#SBATCH --qos=gpushort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=23:50:00
#SBATCH --output=out/pipeline-l2-70b.out
#SBATCH --error=out/pipeline-l2-70b.err

module load python/3.12.3

source venv/bin/activate

export HF_HOME=/p/tmp/maxcall/hf/

# python llm_screen.py meta-llama/Meta-Llama-3.1-8B-Instruct
# python llm_screen.py meta-llama/Meta-Llama-3.1-70B-Instruct
# python llm_screen.py meta-llama/Llama-2-7b-chat-hf
python llm_screen.py meta-llama/Llama-2-70b-chat-hf
