# Start website and open browser
Set-Location $PSScriptRoot\backend
Write-Host "Starting server at http://127.0.0.1:5000" -ForegroundColor Green
Start-Process "http://127.0.0.1:5000"
& .\venv311\Scripts\python.exe app.py
