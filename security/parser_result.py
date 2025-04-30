import json
import argparse
from collections import defaultdict


def main(args):
    res = json.load(open(args.bandit_res_file))
    res.keys()
    files_bandit = defaultdict(list)
    for el in res["results"]:
        file_name = el["filename"].split("/")[-1]
        issue_cwe = el["issue_cwe"]["id"]
        files_bandit[file_name].append(("bandit", issue_cwe))

    # todo
    with open(args.codeql_res_file, "r") as file:
        sarif_data = json.load(file)

    files_codeql = defaultdict(list)

    for run in sarif_data.get("runs", []):
        for result in run.get("results", []):
            # Get the locations of the problem
            for location in result.get("locations", []):
                file_path = (
                    location.get("physicalLocation", {})
                    .get("artifactLocation", {})
                    .get("uri")
                )
                if file_path:
                    files_codeql[file_path].append(("codeql", result["ruleId"]))

    sec_files = defaultdict(list)
    for k, v in files_codeql.items():
        sec_files[k].extend(v)
    for k, v in files_bandit.items():
        sec_files[k].extend(v)

    with open(args.outfile, "w+") as f:
        json.dump(sec_files,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bandit_res_file",
        type=str,
    )
    parser.add_argument(
        "--codeql_res_file",
        type=str,
    )
    parser.add_argument(
        "--outfile",
        type=str,
    )

    args = parser.parse_args()
    main(args)
