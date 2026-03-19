# WEB BASED HYPERTENSION DETECTION USING PPG SIGNALS
Final Year Full Stack AI Project

## Run the website (quick start)

**Option 1 – Double‑click**
- Double‑click **`run.bat`** (Windows).  
  The browser will open at http://127.0.0.1:5000

**Option 2 – Command line**
```bash
cd backend
venv311\Scripts\python.exe app.py
```
Then open http://127.0.0.1:5000 in your browser.

**Option 3 – PowerShell**
```powershell
.\run.ps1
```

## Train ML models (optional, for “Predict with Health Data”)

If the dataset is extracted in `dataset\PPG-BP Database\Data File\`:
```bash
cd backend
python train_models.py
```
Then use **Predict with Health Data** on the website to get predictions from KNN, SVM, and XGBoost.
