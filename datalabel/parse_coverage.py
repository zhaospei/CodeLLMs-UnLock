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
    return start_line, start_test_line, source


import os
def get_coverage(file_coverage,file_source):
    if not os.path.exists(file_coverage):
        return "",""
    if not os.path.exists(file_source):
        return "",""
    
    with open(file_coverage) as f:
        data = json.load(f)
    start_fuc_line, start_test_line,source = get_meta_data(file_source)
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
    return json.dumps(result), source

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