import os
import re
import json
import argparse
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers import AutoTokenizer

from benchmark.MBPP.human_eval.evaluation import evaluate_functional_correctness_and_get_coverage
from .parse_coverage import get_coverage, get_fail_pass_each_testcase, get_spectrum

# Language configuration
language_settings = {
    'python': {'full_name': 'Python', 'indent': 4},
    'cpp': {'full_name': 'cpp', 'indent': 0, 'main': "int main()"},
    'java': {'full_name': 'Java', 'indent': 4, 'main': "public static void main"},
    'cs': {'full_name': "csharp", 'indent': 0, 'main': "public static void Main"},
    'php': {'full_name': "PHP", 'indent': 0},
    'ts': {'full_name': "TypeScript", 'indent': 0},
    'js': {'full_name': "JavaScript", 'indent': 0},
    'sh': {'full_name': "Bash", 'indent': 0}
}

def get_function_name(question: str, lang: str):
    func_lines = [x for x in question.strip().split('\n') if x.strip()]
    
    if lang.lower() == 'python':
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split('(')[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix

    func_name = func_lines[-1].split('{')[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix

def extract_generation_code(gpt_completion: str) -> str:
    try:
        return re.findall(r'```(?:python)?\n(.*?)```', gpt_completion, re.DOTALL | re.IGNORECASE)[0]
    except Exception:
        print("Failed to extract codeblock:\n{}".format(gpt_completion))
        return gpt_completion

def main(args):
    data_root = args.data_root
    continue_from = args.file
    model_name = args.model_name
    language = args.lang
    batch_size = 24
    log_dir = "tmp/mbpp"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    sequences = pd.read_parquet(continue_from).to_dict(orient="records")
    print(f"Loaded {len(sequences)} sequences")
    print(sequences[0])

    results_exec = []
    test_run_results = []
    cleaned_output_results = []
    run_times, run_mem, run_log = [], [], []
    total_samples = []

    for idx in tqdm(range(0, len(sequences), batch_size)):
        log_file = os.path.join(log_dir, f'{model_name.replace("/", "_")}_{idx}_shot_log_{language}.json')
        
        with open(log_file, "w") as tmpfile:
            batch_lines = []
            for sequence in sequences[idx:idx + batch_size]:
                suffix_prediction = extract_generation_code(sequence['generation'])
                res = {
                    "task_id": sequence['task_id'],
                    "generation": suffix_prediction,
                    "prompt": '',
                    "completion_id": sequence['completion_id']
                }
                batch_lines.append(res)
                tmpfile.write(json.dumps(res) + "\n")
                tmpfile.flush()

            results = evaluate_functional_correctness_and_get_coverage(
                input_file=log_file,
                problem_file=os.path.join(data_root, "mbpp_test.jsonl"),
                tmp_dir=log_dir,
                language=language,
                n_workers=4,
                is_mbpp=True
            )

            results_exec.append(results)

            for line in batch_lines:
                cid = line["completion_id"]
                cleaned_output_results.append(line["generation"])
                test_run_results.append(results[cid][0][1]["passed"])
                run_times.append(results[cid][0][1]["execution_time"])
                run_mem.append(results[cid][0][1]["memory"])
                run_log.append(results[cid][0][1]["result"])
                total_samples.append(line)

    with open(continue_from.replace(".parquet", ".json"), "w") as f:
        json.dump(results_exec, f)

    # Collect detailed results
    results_df = pd.DataFrame(sequences)
    results_df["label"] = test_run_results
    results_df["memory"] = run_mem
    results_df["time"] = run_times
    results_df["cleaned_code"] = cleaned_output_results

    results_df["run_log"] = run_log
    results_df["test_code"] = test_codes
    coverage_list, source_list, test_codes = [], [], []
    for _, row in results_df.iterrows():
        coverage_file = f'tmp_dir/coverage/mbpp_coverage_{row["task_id"]}_{row["completion_id"]}.json'
        source_file = f'tmp_dir/source/test_{row["task_id"]}_{row["completion_id"]}.py'
        coverage_dict, source, test_code = get_coverage(coverage_file, source_file)
        coverage_list.append(coverage_dict)
        source_list.append(source)
        test_codes.append(test_code)

    
    result_test, test_faileds, total_of_test = [], [], []
    for _, row in tqdm(results_df.iterrows()):
        result, number_of_test, test_failed = get_fail_pass_each_testcase(row["run_log"])
        result_test.append(result)
        test_faileds.append(" ".join(test_failed))
        total_of_test.append(number_of_test)

    results_df["coverage"] = coverage_list
    results_df["source_coverage"] = source_list
    results_df["result_test"] = result_test
    results_df["test_failed"] = test_faileds
    results_df["total_of_test"] = total_of_test

    spectrums = [get_spectrum(row) for _, row in tqdm(results_df.iterrows())]
    results_df["spectrum"] = spectrums
    results_df.to_parquet(continue_from.replace(".parquet", "_label.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    args = parser.parse_args()
    main(args)
