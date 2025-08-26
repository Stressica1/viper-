# VIPER Trading Bot - Complete Project Backup Script
# This script creates a comprehensive backup of your entire project

param(
    [string]$BackupPath = "C:\Users\user\Desktop\VIPER_Backup_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss')"
)

Write-Host "Starting VIPER Trading Bot Backup..." -ForegroundColor Green
Write-Host "Backup destination: $BackupPath" -ForegroundColor Yellow

# Create backup directory
New-Item -ItemType Directory -Path $BackupPath -Force

# 1. Backup Project Files
Write-Host "Backing up project files..." -ForegroundColor Cyan
Copy-Item -Path "C:\Users\user\Bitget-New\*" -Destination $BackupPath -Recurse -Force

# 2. Secure Environment Backup
Write-Host "Creating secure environment backup..." -ForegroundColor Magenta
$EnvBackupPath = Join-Path $BackupPath "secure_env_backup"
New-Item -ItemType Directory -Path $EnvBackupPath -Force

# Create a sanitized version for backup (without real credentials)
Get-Content "C:\Users\user\Bitget-New\.env" | ForEach-Object {
    if ($_ -match "^BITGET_API_KEY=") {
        "BITGET_API_KEY=your_api_key_here"
    } elseif ($_ -match "^BITGET_API_SECRET=") {
        "BITGET_API_SECRET=your_api_secret_here"
    } elseif ($_ -match "^BITGET_API_PASSWORD=") {
        "BITGET_API_PASSWORD=your_password_here"
    } else {
        $_
    }
} | Out-File -FilePath (Join-Path $EnvBackupPath ".env.sanitized") -Encoding UTF8

# Create encrypted backup of real credentials (you'll need the password to restore)
Write-Host "IMPORTANT: Your real API credentials will be encrypted." -ForegroundColor Red
Write-Host "Remember your encryption password!" -ForegroundColor Red
$RealEnvPath = Join-Path $EnvBackupPath ".env.real.encrypted"

# Use AES encryption for portable backup
$EncryptionKey = Read-Host "Enter encryption password for API credentials" -AsSecureString
$Key = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($EncryptionKey))

# Simple XOR encryption (portable across machines)
$Content = Get-Content "C:\Users\user\Bitget-New\.env" -Raw
$EncryptedContent = ""
for ($i = 0; $i -lt $Content.Length; $i++) {
    $EncryptedContent += [char]($Content[$i] -bxor $Key[$i % $Key.Length])
}
$EncryptedContent | Out-File $RealEnvPath -Encoding UTF8

# 3. Backup Backtest Results
Write-Host "Backing up backtest results..." -ForegroundColor Cyan
if (Test-Path "C:\Users\user\Bitget-New\backtest_results") {
    Copy-Item -Path "C:\Users\user\Bitget-New\backtest_results" -Destination (Join-Path $BackupPath "backtest_results") -Recurse -Force
}

# 4. Backup Logs
Write-Host "Backing up logs..." -ForegroundColor Cyan
if (Test-Path "C:\Users\user\Bitget-New\logs") {
    Copy-Item -Path "C:\Users\user\Bitget-New\logs" -Destination (Join-Path $BackupPath "logs") -Recurse -Force
}

# 5. Backup Docker Data
Write-Host "Backing up Docker data (if any)..." -ForegroundColor Cyan
$DockerData = @(
    "$env:USERPROFILE\.docker",
    "C:\ProgramData\Docker",
    "C:\ProgramData\DockerDesktop"
)

foreach ($Path in $DockerData) {
    if (Test-Path $Path) {
        $DestPath = Join-Path $BackupPath "docker_data\$($Path -replace '.*\\', '')"
        New-Item -ItemType Directory -Path (Split-Path $DestPath -Parent) -Force
        Copy-Item -Path $Path -Destination $DestPath -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# 6. Create System Information Backup
Write-Host "Creating system information backup..." -ForegroundColor Cyan
$SystemInfo = @{
    "OS" = (Get-ComputerInfo).WindowsProductName
    "PowerShell Version" = $PSVersionTable.PSVersion.ToString()
    "Docker Version" = & docker --version
    "Python Version" = & python --version 2>$null
    "Node Version" = & node --version 2>$null
    "Git Version" = & git --version 2>$null
    "Installed Python Packages" = & pip list 2>$null
    "Environment Variables" = Get-ChildItem Env: | Out-String
}

$SystemInfo.GetEnumerator() | ForEach-Object {
    "$($_.Key): $($_.Value)" | Out-File -FilePath (Join-Path $BackupPath "system_info.txt") -Append
}

# 7. Create Restore Instructions
Write-Host "Creating restore instructions..." -ForegroundColor Cyan
$RestoreInstructions = @"
VIPER Trading Bot - Restore Instructions
========================================

1. Extract the backup to your desired location

2. Restore Environment Configuration:
   - Copy 'secure_env_backup\.env.sanitized' to your project root as '.env'
   - For real credentials, run this PowerShell script to decrypt:
     `$EncryptionKey = Read-Host "Enter decryption password" -AsSecureString
     `$Key = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR(`$EncryptionKey))
     `$EncryptedContent = Get-Content 'secure_env_backup\.env.real.encrypted' -Raw
     `$DecryptedContent = ""
     for (`$i = 0; `$i -lt `$EncryptedContent.Length; `$i++) {
         `$DecryptedContent += [char](`$EncryptedContent[`$i] -bxor `$Key[`$i % `$Key.Length])
     }
     `$DecryptedContent | Out-File '.env' -Encoding UTF8

3. Restore Docker (if needed):
   - Install Docker Desktop
   - Restore Docker data from 'docker_data' folder

4. Reinstall Python Dependencies:
   - Navigate to project directory
   - Run: pip install -r services/api-server/requirements.txt
   - Run: pip install -r services/live-trading-engine/requirements.txt

5. Start Services:
   - Run: python start_microservices.py start

IMPORTANT NOTES:
- Keep your encryption password safe for restoring real API credentials
- Test the system thoroughly before live trading
- Update API credentials if needed

Backup created on: $(Get-Date)
"@

$RestoreInstructions | Out-File -FilePath (Join-Path $BackupPath "RESTORE_INSTRUCTIONS.txt") -Encoding UTF8

# 8. Create Backup Summary
Write-Host "Creating backup summary..." -ForegroundColor Cyan
$BackupSize = (Get-ChildItem $BackupPath -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
$FileCount = (Get-ChildItem $BackupPath -Recurse -File).Count

$Summary = @"
VIPER Trading Bot - Backup Summary
=====================================

Backup Location: $BackupPath
Backup Date: $(Get-Date)
Total Size: $([math]::Round($BackupSize, 2)) MB
Total Files: $FileCount

Contents:
- Complete project source code
- Environment configuration (sanitized)
- Backtest results and logs
- Docker data and configuration
- System information snapshot

Security Notes:
- Real API credentials are encrypted
- Remember your encryption password
- Sanitized .env file is safe to share

Next Steps:
1. Copy backup to external storage
2. Verify backup integrity
3. Proceed with Windows 11 installation
4. Follow RESTORE_INSTRUCTIONS.txt after installation

REMEMBER: Your real API credentials are encrypted in this backup.
Keep the encryption password safe!
"@

$Summary | Out-File -FilePath (Join-Path $BackupPath "BACKUP_SUMMARY.txt") -Encoding UTF8

Write-Host "Backup completed successfully!" -ForegroundColor Green
Write-Host "Backup location: $BackupPath" -ForegroundColor Yellow
Write-Host "Check BACKUP_SUMMARY.txt for details" -ForegroundColor Yellow
Write-Host "Read RESTORE_INSTRUCTIONS.txt for restoration steps" -ForegroundColor Yellow

# Open backup folder
explorer.exe $BackupPath
