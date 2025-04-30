# python3 -m pipeline.save_hidden_states2 --model deepseek-ai/deepseek-coder-6.7b-instruct --dataset human_eval --num_generations_per_prompt 10  --project_ind 0 --layer 0 1 4 8 12 16 20 24 --max_new_tokens 512 --language python --max_num_gen_once 5
MODEL="deepseek-ai/deepseek-coder-6.7b-instruct"
OUT_DIR="output/deepseek-ai_deepseek-coder-6.7b-instruct_human_eval_python_0_1_4_8_12_16_20_24_28_32/temp2"
python3 -m pipeline.extract_token_code2 --model $MODEL --dataset security --layers 0 1 4 8 12 16 20 24 28 32 --generate_dir $OUT_DIR --type LFCLF

MODEL="codellama/CodeLlama-7b-Instruct-hf"
OUT_DIR="output/codellama_CodeLlama-7b-Instruct-hf_human_eval_python_0_1_4_8_12_16_20_24_28_32/temp2"
python3 -m pipeline.extract_token_code2 --model $MODEL --dataset security --layers 0 1 4 8 12 16 20 24 28 32 --generate_dir $OUT_DIR --type LFCLF

MODEL="ise-uiuc/Magicoder-S-DS-6.7B"
OUT_DIR="output/ise-uiuc_Magicoder-S-DS-6.7B_human_eval_python_0_1_4_8_12_16_20_24_28_32/temp2"
python3 -m pipeline.extract_token_code2 --model $MODEL --dataset security --layers 0 1 4 8 12 16 20 24 28 32 --generate_dir $OUT_DIR --type LFCLF
