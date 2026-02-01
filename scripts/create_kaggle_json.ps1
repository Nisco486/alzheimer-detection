<#
.SYNOPSIS
Creates C:\Users\<user>\.kaggle\kaggle.json from a project `.env` file or prompts for credentials.

Usage:
  .\scripts\create_kaggle_json.ps1                # reads app\.env by default
  .\scripts\create_kaggle_json.ps1 -EnvPath .\app\.env

#>
param(
    [string]$EnvPath = ".\app\.env"
)

function Get-CredFromEnvFile {
    param($path)
    if (-Not (Test-Path $path)) { return $null }
    $lines = Get-Content $path
    $user = ($lines | Where-Object { $_ -match '^KAGGLE_USERNAME=' }) -replace '^KAGGLE_USERNAME='
    $key  = ($lines | Where-Object { $_ -match '^KAGGLE_KEY=' }) -replace '^KAGGLE_KEY='
    if ($user) { $user = $user.Trim() }
    if ($key)  { $key  = $key.Trim() }
    if ($user -and $key) { return @{username=$user; key=$key} }
    return $null
}

Write-Host "Reading credentials from '$EnvPath'..."
$creds = Get-CredFromEnvFile -path $EnvPath
if (-not $creds) {
    Write-Host "Could not parse credentials from $EnvPath." -ForegroundColor Yellow
    $username = Read-Host "Enter Kaggle username"
    $key = Read-Host "Enter Kaggle key (will be visible)"
    $creds = @{username=$username; key=$key}
}

$kaggleDir = Join-Path $env:USERPROFILE '.kaggle'
New-Item -ItemType Directory -Path $kaggleDir -Force | Out-Null

$jsonPath = Join-Path $kaggleDir 'kaggle.json'
$json = @{username=$creds.username; key=$creds.key} | ConvertTo-Json -Compress
Set-Content -Path $jsonPath -Value $json -NoNewline -Encoding UTF8

# Restrict permissions to the current user (remove inherited permissions)
try {
    icacls $jsonPath /inheritance:r /grant:r "$($env:USERNAME):(R)" | Out-Null
    Write-Host "Wrote and secured: $jsonPath" -ForegroundColor Green
} catch {
    Write-Host "Warning: failed to change file permissions. You can set them manually." -ForegroundColor Yellow
}

Write-Host "Contents:" -ForegroundColor Cyan
Get-Content $jsonPath | Write-Host

Write-Host "You can verify with: kaggle config view" -ForegroundColor Cyan
