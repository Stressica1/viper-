# VIPER Trading Bot GitHub Push Script
Write-Host "🚀 Pushing VIPER Trading Bot to GitHub..." -ForegroundColor Green
Write-Host ""

try {
    # Check git status
    Write-Host "Checking git status..." -ForegroundColor Yellow
    git status

    # Add remote if it doesn't exist
    Write-Host "`nSetting up remote repository..." -ForegroundColor Yellow
    $remoteExists = git remote get-url origin 2>$null
    if (-not $remoteExists) {
        git remote add origin https://github.com/Stressica1/viper-.git
        Write-Host "✅ Remote added successfully" -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Remote already exists" -ForegroundColor Blue
    }

    # Set remote with credentials
    Write-Host "`nSetting remote URL with authentication..." -ForegroundColor Yellow
    git remote set-url origin https://github.com/Stressica1/viper-.git

    # Push to GitHub
    Write-Host "`n🚀 Pushing to GitHub..." -ForegroundColor Yellow
    git push -u origin main

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ SUCCESS! VIPER Trading Bot pushed to GitHub!" -ForegroundColor Green
        Write-Host "🌐 Repository: https://github.com/Stressica1/viper-" -ForegroundColor Cyan
        Write-Host "`n🎉 Your production-ready trading system is now on GitHub!" -ForegroundColor Magenta
    } else {
        Write-Host "`n❌ Push failed. Please check the error message above." -ForegroundColor Red
        Write-Host "`n💡 Alternative: Use GitHub Desktop or upload files manually" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "`n❌ Error occurred: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "`n💡 Try using GitHub Desktop for manual upload" -ForegroundColor Yellow
}

Write-Host "`nPress Enter to continue..." -ForegroundColor Gray
Read-Host
