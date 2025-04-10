import json
from collections import defaultdict


def get_meta_data(source_file):
    with open(source_file) as ff:
        content = ff.read()
        start_line = -1
        start_test_line = -1
        contentlines = content.splitlines()
        for idx,l in enumerate(contentlines):
            if l.startswith('def ') and start_line < 0:
                start_line=idx 
            if l.startswith('def test_0') and start_test_line < 0:
                start_test_line = idx 
            if start_line > 0 and start_test_line >0:
                break
        source = '\n'.join(contentlines[start_line:start_test_line-1])
    return start_line, start_test_line, source, content


import os
def get_coverage(file_coverage,file_source):
    if not os.path.exists(file_coverage):
        return "","", ""
    if not os.path.exists(file_source):
        return "","", ""
    
    with open(file_coverage) as f:
        data = json.load(f)
    start_fuc_line, start_test_line,source,test_code = get_meta_data(file_source)
    funcs = list(data['files'].values())[0]['functions']
    funcs = list(funcs.values())
    executed_lines = funcs[0]['executed_lines']
    result = defaultdict(list)
    funcs[0]['contexts']
    for k,v in funcs[0]['contexts'].items():
        k = int(k)
        if k not in executed_lines:
            continue
        for test_case in v:
            test_name = test_case.split('.')[-1]
            if len(test_name) <= 1:
                continue
            result[test_name].append(k-start_fuc_line)
    return json.dumps(result), source, test_code

import re
def get_fail_pass_each_testcase(log):
    log_lines = log.splitlines()
    matches = re.findall(r'(\d+)\s+failed|(\d+)\s+passed', log_lines[-1])
    results = [int(num) for pair in matches for num in pair if num]
    number_of_test = sum(results)
    if "failed" in log_lines[-1]:
        index = -1
        for idx,line in enumerate(log_lines):
            if "short test summary info" in line:
                index = idx
                break
        if index == -1:
            print("BUG")
        result = list()
        for line in log_lines[index+1:-1]:
            line_split = line.split()
            test_case_result = line_split[0]
            test_case_name = line.split(".py::")[-1]
            if len(test_case_name.split()) > 0:
                test_case_name = test_case_name.split()[0]
            if test_case_result == "FAILED":
                result.append(test_case_name)
        return "Failed",number_of_test, result
    elif "error" in log_lines[-1]:
        return "Error",number_of_test, []
    elif "timed out" in log_lines[-1]:
        return "Time out",number_of_test, []
    else:
        return "Passed",number_of_test, []
    
    
import json 
import math

def compute_tarantula(coverage_matrix, test_results):
    if len(coverage_matrix) <= 0:
        return []
    num_tests = len(coverage_matrix)
    num_statements = len(coverage_matrix[0])
    failed_tests = [i for i, result in enumerate(test_results) if not result]
    passed_tests = [i for i, result in enumerate(test_results) if result]
    total_failed = len(failed_tests)
    total_passed = len(passed_tests)
    scores = []
    for stmt in range(num_statements):
        ef = sum(coverage_matrix[i][stmt] for i in failed_tests)  # executed & failed
        ep = sum(coverage_matrix[i][stmt] for i in passed_tests)  # executed & passed

        if ef == 0 and ep == 0:
            score = 0.0  # not executed at all
        elif (ef + ep) == 0:
            score = 0.0
        else:
            fail_ratio = ef / total_failed if total_failed > 0 else 0
            pass_ratio = ep / total_passed if total_passed > 0 else 0
            denominator = fail_ratio + pass_ratio
            score = fail_ratio / denominator if denominator > 0 else 0.0

        scores.append(score)

    return scores


def get_matrixes(coverage,test_failed):
    #TODO: not theo boi thu tu 
    coverage_dict = json.loads(coverage)
    all_statements = sorted(set(stmt for stmts in coverage_dict.values() for stmt in stmts))
    coverage_matrix = list()
    test_results = list()
    for test_name,cov in coverage_dict.items():
        tmp_cov = list()
        for line in all_statements:
            if line in cov:
                tmp_cov.append(1)
            else:
                tmp_cov.append(0)
        coverage_matrix.append(tmp_cov)
        test_results.append(test_name not in test_failed)
    return coverage_matrix, test_results, all_statements

def get_spectrum(row):
    if row['result_test'] != "Failed":
        return ''
    if len(row['coverage']) <= 0:
        return ''
    converage_matrix, test_results, all_statements = get_matrixes(row['coverage'],row['test_failed'])
    scores = compute_tarantula(converage_matrix,test_results)
    spectrum = dict()
    if len(all_statements) <= 0:
        return ''
    for line,score in zip(all_statements,scores):
        spectrum[line] = score 
    return json.dumps(spectrum)