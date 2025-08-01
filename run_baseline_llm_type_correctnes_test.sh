MODEL=deepseek-ai/deepseek-coder-1.3b-instruct
SOURCE_FILE=zero-shot/ds13_security.parquet
python3 pipeline/baseline_generation_correctness_type.py \
 --model $MODEL \
 --source_file $SOURCE_FILE \
 --prompt_file ${SOURCE_FILE}.json \
 --problem_file dataset_sec.jsonl \
 --type sec