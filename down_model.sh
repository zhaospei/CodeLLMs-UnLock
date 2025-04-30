pip install hf_transfer
pip install "huggingface_hub[hf_transfer]"
MODEL='deepseek-ai/deepseek-coder-6.7b-instruct'
MODEL='codellama/CodeLlama-7b-Instruct-hf'
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $MODEL

MODEL='ise-uiuc/Magicoder-DS-6.7B'

HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download $MODEL
