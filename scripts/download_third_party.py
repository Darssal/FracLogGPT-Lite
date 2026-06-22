#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPOS = {
    "ms-swift": "https://github.com/modelscope/ms-swift.git",
    "SimPO": "https://github.com/princeton-nlp/SimPO.git",
    "VCD": "https://github.com/DAMO-NLP-SG/VCD.git",
    "transformers": "https://github.com/huggingface/transformers.git",
    "peft": "https://github.com/huggingface/peft.git",
    "vllm": "https://github.com/vllm-project/vllm.git",
}

ARCHIVES = {
    "ms-swift": ("modelscope", "ms-swift"),
    "SimPO": ("princeton-nlp", "SimPO"),
    "VCD": ("DAMO-NLP-SG", "VCD"),
    "transformers": ("huggingface", "transformers"),
    "peft": ("huggingface", "peft"),
    "vllm": ("vllm-project", "vllm"),
}


def run(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def remote_head(url: str) -> str | None:
    try:
        output = run(["git", "ls-remote", url, "HEAD"])
    except Exception:
        return None
    first = output.split()[0] if output.split() else ""
    return first or None


def download_archive(name: str, third_party: Path) -> dict:
    owner, repo = ARCHIVES[name]
    target = third_party / name
    if target.exists() and any(target.iterdir()):
        url = REPOS[name]
        return {"name": name, "url": url, "path": str(target), "commit": remote_head(url), "status": "exists_non_git"}

    tmp_zip = third_party / f"{name}.zip"
    extract_dir = third_party / f".extract_{name}"
    for branch in ["main", "master"]:
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FracLogExperimentHarness"})
            with urllib.request.urlopen(request, timeout=180) as response:
                tmp_zip.write_bytes(response.read())
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True)
            with zipfile.ZipFile(tmp_zip) as archive:
                archive.extractall(extract_dir)
            roots = [path for path in extract_dir.iterdir() if path.is_dir()]
            if not roots:
                raise RuntimeError(f"Archive {url} had no root directory")
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(roots[0]), str(target))
            source = {
                "name": name,
                "url": url,
                "path": str(target),
                "branch": branch,
                "commit": remote_head(REPOS[name]),
                "status": "archive_downloaded",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "version_note": "Downloaded from GitHub branch archive because git clone could not reach github.com. Commit SHA unavailable without GitHub API or git access.",
            }
            (target / ".fraclog_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
            return source
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, RuntimeError, zipfile.BadZipFile):
            continue
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
    raise RuntimeError(f"Could not download GitHub archive for {name}")


def clone_or_update(name: str, url: str, third_party: Path, depth: int) -> dict:
    target = third_party / name
    if target.exists():
        if (target / ".git").exists():
            status = "exists"
        elif any(target.iterdir()):
            return {"name": name, "url": url, "path": str(target), "commit": remote_head(url), "status": "exists_non_git"}
        else:
            target.rmdir()
            status = "missing"
    else:
        status = "missing"
    if status == "missing":
        command = ["git", "clone"]
        if depth > 0:
            command += ["--depth", str(depth)]
        command += [url, str(target)]
        run(command)
        status = "cloned"
    commit = run(["git", "rev-parse", "HEAD"], cwd=target)
    remote = run(["git", "remote", "get-url", "origin"], cwd=target)
    return {"name": name, "url": remote, "path": str(target), "commit": commit, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download third-party GitHub projects and record versions.")
    parser.add_argument("--third-party-dir", default="third_party")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--depth", type=int, default=1, help="Use 0 for full git history; default keeps full current project files with shallow history.")
    args = parser.parse_args()

    third_party = Path(args.third_party_dir)
    reports = Path(args.reports_dir)
    third_party.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    versions = []
    failures = []
    for name, url in REPOS.items():
        try:
            versions.append(clone_or_update(name, url, third_party, args.depth))
        except Exception as exc:
            try:
                versions.append(download_archive(name, third_party))
            except Exception as archive_exc:
                failures.append({"name": name, "url": url, "error": str(exc), "archive_error": str(archive_exc)})

    payload = {"repos": versions, "failures": failures}
    (reports / "third_party_versions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Third-Party Versions", ""]
    for row in versions:
        lines.append(f"- {row['name']}: `{row['commit']}` ({row['status']})")
    if failures:
        lines.extend(["", "## Failures"])
        for row in failures:
            lines.append(f"- {row['name']}: {row['error']}")
    (reports / "third_party_versions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
