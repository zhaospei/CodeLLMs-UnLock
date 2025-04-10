#done sh cs js java python php 
#error ts cpp 
#ndone  
# for L in "cs" "java" "js" "php" "python" "sh" "ts" 
# do 
# LANG=$L
# python -m datalabel.label_human_eval_instruct2 \
# --data_root benchmark/HumanEval/data \
# --file data/ds67$LANG/output2/LFCLF_embedding_human_eval_deepseek-ai_deepseek-coder-6.7b-instruct_1.parquet \
# --model_name deepseek-ai/deepseek-coder-6.7b-instruct \
# --lang $LANG
# done

python -m datalabel.label_mbpp_instruct2 \
--data_root benchmark/MBPP/data \
--file data/LFCLF_embedding_mbpp_codellama_CodeLlama-7b-Instruct-hf_32_label.parquet  \
--model_name codellama/CodeLlama-7b-Instruct-hf \
--lang python \
--working_dir codellama_mbpp