"""Compare the exact published site tree, not a skipped workflow's success."""
from __future__ import annotations

import json
import os
import re
import subprocess


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def actual_deployed_tree(jobs, legacy_sha, resolve_tree):
    for job in jobs:
        steps = job.get("steps") or []
        if not any(s["name"] == "Deploy to GitHub Pages" and
                   s.get("conclusion") == "success" for s in steps):
            continue
        for step in steps:
            match = re.fullmatch(r"Record deployed site tree ([0-9a-f]{40})", step["name"])
            if match and step.get("conclusion") == "success":
                return match.group(1)
        # Legacy successful runs checked out main; their head SHA is the best
        # available reference. For new deployments the recorded tree is exact.
        try:
            return resolve_tree(legacy_sha)
        except subprocess.CalledProcessError:
            return None
    return None


def last_deployed_tree(repository, current_run, query=api, resolve_tree=None):
    if resolve_tree is None:
        resolve_tree = lambda sha: git("rev-parse", sha + ":site")
    prefix = f"repos/{repository}/actions"
    runs = query(prefix + "/workflows/pages.yml/runs?status=success&per_page=30")["workflow_runs"]
    for run in runs:
        if str(run["id"]) == str(current_run):
            continue
        jobs = query(prefix + f"/runs/{run['id']}/jobs?per_page=100")["jobs"]
        tree = actual_deployed_tree(jobs, run["head_sha"], resolve_tree)
        if tree:
            return tree
    return None


def main():
    sha = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD:site")
    force = os.environ.get("FORCE_DEPLOY", "").lower() == "true"
    previous = None if force else last_deployed_tree(os.environ["GITHUB_REPOSITORY"],
                                                     os.environ["GITHUB_RUN_ID"])
    deploy = force or tree != previous
    print("Current site tree:", tree, "; last successfully deployed:", previous,
          "; deploy:", deploy)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        out.write(f"sha={sha}\ntree={tree}\ndeploy={str(deploy).lower()}\n")


if __name__ == "__main__":
    main()
