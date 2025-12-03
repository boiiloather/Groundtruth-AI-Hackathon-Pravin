# AI Creative Studio - API Keys Setup Script
# Run this script to set your API keys for the current PowerShell session

Write-Host "Setting API Keys for AI Creative Studio..." -ForegroundColor Green
Write-Host ""

# Groq API Key (FREE - Recommended)
$groqKey = Read-Host "Enter your Groq API Key (or press Enter to skip)"
if ($groqKey) {
    $env:GROQ_API_KEY = $groqKey
    Write-Host "✓ Groq API Key set" -ForegroundColor Green
} else {
    Write-Host "⚠ Groq API Key skipped" -ForegroundColor Yellow
}

Write-Host ""

# Hugging Face API Key (FREE - Recommended)
$hfKey = Read-Host "Enter your Hugging Face API Token (or press Enter to skip)"
if ($hfKey) {
    $env:HUGGINGFACE_API_KEY = $hfKey
    Write-Host "✓ Hugging Face API Key set" -ForegroundColor Green
} else {
    Write-Host "⚠ Hugging Face API Key skipped" -ForegroundColor Yellow
}

Write-Host ""

# OpenAI API Key (Optional - if you have free credits)
$openaiKey = Read-Host "Enter your OpenAI API Key (or press Enter to skip)"
if ($openaiKey) {
    $env:OPENAI_API_KEY = $openaiKey
    Write-Host "✓ OpenAI API Key set" -ForegroundColor Green
} else {
    Write-Host "⚠ OpenAI API Key skipped" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "API Keys configured! Now run: python app.py" -ForegroundColor Cyan
Write-Host ""

