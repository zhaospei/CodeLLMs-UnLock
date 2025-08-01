# export PYTHONPATH=`pwd`
# pip install -r req.txt
#benchmark/MBPP/data/mbpp_test.jsonl
#benchmark/HumanEval/data/humaneval-python.jsonl
#dataset_sec.jsonl
MODEL=google/codegemma-7b-it
SOURCE_FILE=zero-shot/cg70_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
MODEL=deepseek-ai/deepseek-coder-6.7b-instruct
SOURCE_FILE=zero-shot/ds67_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
MODEL=deepseek-ai/deepseek-coder-1.3b-instruct
SOURCE_FILE=zero-shot/ds13_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
MODEL=ise-uiuc/Magicoder-S-DS-6.7B
SOURCE_FILE=zero-shot/mc67_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
MODEL=codellama/CodeLlama-7b-Instruct-hf
SOURCE_FILE=zero-shot/cl70_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
MODEL=codellama/CodeLlama-13b-Instruct-hf
SOURCE_FILE=zero-shot/cl13_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec
# =====
