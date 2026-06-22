param(
  [string]$RepoName = "FracLogGPT-Lite",
  [ValidateSet("private", "public", "internal")]
  [string]$Visibility = "private",
  [string]$RemoteName = "origin",
  [string]$ProxyUrl = ""
)

$ErrorActionPreference = "Stop"

if ($ProxyUrl) {
  $env:HTTP_PROXY = $ProxyUrl
  $env:HTTPS_PROXY = $ProxyUrl
  $env:ALL_PROXY = $ProxyUrl
}

$ghCommand = Get-Command gh -ErrorAction SilentlyContinue
if ($ghCommand) {
  $gh = $ghCommand.Source
} else {
  $candidates = @(
    "C:\Program Files\GitHub CLI\gh.exe",
    "C:\Program Files (x86)\GitHub CLI\gh.exe",
    "$env:LOCALAPPDATA\Programs\GitHub CLI\gh.exe"
  )
  $gh = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}

if (-not $gh) {
  throw "GitHub CLI 'gh' is not installed. Install it with: winget install --id GitHub.cli -e"
}

& $gh auth status *> $null
if ($LASTEXITCODE -ne 0) {
  throw "GitHub CLI is not authenticated. Run: gh auth login"
}

$owner = (& $gh api user --jq ".login").Trim()
if (-not $owner) {
  throw "Could not determine the authenticated GitHub user."
}

$repoFullName = "$owner/$RepoName"
$remoteUrl = "https://github.com/$repoFullName.git"
$visibilityArg = switch ($Visibility) {
  "private" { "--private" }
  "public" { "--public" }
  "internal" { "--internal" }
}

$repoExists = $false
try {
  $existingRepo = (& $gh repo view $repoFullName --json nameWithOwner --jq ".nameWithOwner" 2>$null).Trim()
  if ($LASTEXITCODE -eq 0 -and $existingRepo) {
    $repoExists = $true
  }
} catch {
  $repoExists = $false
}

if (-not $repoExists) {
  & $gh repo create $RepoName $visibilityArg --source "." --remote $RemoteName --push
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to create and push GitHub repository $repoFullName."
  }
  Write-Host "Created and pushed https://github.com/$repoFullName"
  exit 0
}

& git remote get-url $RemoteName *> $null
if ($LASTEXITCODE -eq 0) {
  & git remote set-url $RemoteName $remoteUrl
} else {
  & git remote add $RemoteName $remoteUrl
}

& git push -u $RemoteName main
if ($LASTEXITCODE -ne 0) {
  throw "Push failed. If $repoFullName already contains commits, inspect it before force-pushing."
}

Write-Host "Pushed https://github.com/$repoFullName"
