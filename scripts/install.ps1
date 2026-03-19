# Install rawq — downloads the latest release binary and adds it to PATH.
# Usage: irm https://raw.githubusercontent.com/auyelbekov/rawq/main/scripts/install.ps1 | iex

$ErrorActionPreference = "Stop"

$Repo = "auyelbekov/rawq"
$InstallDir = if ($env:RAWQ_INSTALL_DIR) { $env:RAWQ_INSTALL_DIR } else { "$env:LOCALAPPDATA\rawq\bin" }
$Archive = "rawq-windows-x86_64.zip"

# Get latest release tag
$Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
$Tag = $Release.tag_name
$Url = "https://github.com/$Repo/releases/download/$Tag/$Archive"

Write-Host "Installing rawq $Tag for Windows x86_64..."
Write-Host "  From: $Url"
Write-Host "  To:   $InstallDir\rawq.exe"

# Download
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$TempZip = "$env:TEMP\rawq-download.zip"
Invoke-WebRequest -Uri $Url -OutFile $TempZip

# Extract
Expand-Archive -Path $TempZip -DestinationPath $InstallDir -Force
Remove-Item $TempZip

# Add to PATH if not already there
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$InstallDir;$UserPath", "User")
    Write-Host ""
    Write-Host "Added $InstallDir to your PATH. Restart your terminal to use rawq."
}

Write-Host ""
Write-Host "rawq $Tag installed successfully."
& "$InstallDir\rawq.exe" --version
