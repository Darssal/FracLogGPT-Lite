# Security and Data Policy

Do not open public issues or pull requests that include private well-log images, annotations, checkpoints, credentials, local paths, unpublished manuscript text, or reviewer/author information.

Before publishing a fork or branch, run a sensitive-content scan over tracked files:

```powershell
rg -n -i "token|secret|password|api[_-]?key|ghp_|gho_|github_pat_|hf_|sk-|C:\\\\Users|D:\\\\program" .
git status -sb
git ls-files
```

If sensitive content was committed, remove it from the working tree and rewrite the public history before changing repository visibility to public.
