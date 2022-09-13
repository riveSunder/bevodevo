import os


if __name__ == "__main__":

    test_command = "coverage run -m bevodevo.tests.test_all"
    coverage_command = "coverage report -m > coverage.txt"

    git_command = "git add coverage.txt"

    print(test_command)
    print(coverage_command)
    
    os.system(test_command)
    os.system(coverage_command)

    with open("coverage.txt", "r") as f:
        for line in f.readlines():
            if "TOTAL" in line:
                summary = line

    commit_command = f"git commit -m 'test commit summary: {summary}'"

    print(git_command)
    print(commit_command)

    print(f"***{summary}***")

    os.system(git_command)
    os.system(commit_command)
