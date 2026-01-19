# Setup Guide for Recruiters & Reviewers

This guide helps you run the project locally in under 5 minutes.

## Step 1: Prerequisites Check

Make sure you have:
- **Python 3.8 or higher** ([Download here](https://www.python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/downloads))

Verify installation:
```bash
python --version  # Should show 3.8+
git --version
```

---

## Step 2: Clone & Install

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-analytics-platform.git
cd customer-analytics-platform

# Install dependencies
pip install -r requirements.txt
```

**Note**: If you see "command not found", use `python3` and `pip3` instead.

---

## Step 3: Generate Data

```bash
python src/database_setup.py
```

**What this does**:
- Downloads the UCI Online Retail dataset (20 MB)
- Cleans the data (removes cancellations, nulls)
- Creates `data/customer_analytics.db` (SQLite database)

**Expected output**:
```
Downloading data from UCI...
Download complete.
Loading Excel file...
Raw data shape: (541909, 8)
Cleaning data...
Cleaned data shape: (397924, 9)
Ingesting into SQLite...
Ingestion complete.
```

---

## Step 4: Train the ML Model

```bash
python src/model.py
```

**What this does**:
- Engineers features (RFM + Tenure)
- Trains a Random Forest classifier
- Saves `data/churn_model.pkl`

**Expected output**:
```
Engineering features...
Training snapshot date: 2011-09-10
Target distribution:
is_churn
0    1921
1    1449
Training model...
Model Evaluation:
              precision    recall  f1-score
           0       0.67      0.70      0.69
           1       0.57      0.53      0.55
    accuracy                           0.63
Saving model to data/churn_model.pkl
```

---

## Step 5: Launch the Dashboard

```bash
streamlit run app.py
```

**What this does**:
- Starts a local web server
- Opens your browser to `http://localhost:8501`

**Expected output**:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you installed dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "File not found: data/customer_analytics.db"
**Solution**: Run the data pipeline first:
```bash
python src/database_setup.py
```

### Issue: Dashboard shows errors
**Solution**: Make sure you trained the model:
```bash
python src/model.py
```

### Issue: Path with spaces (e.g., "New folder")
**Solution**: Use quotes in PowerShell:
```powershell
cd "C:\projects\DE\Customer-Analysis\customer_analytics_platform"
```

---

## What to Explore

Once the dashboard is running:

1. **Executive Summary** → See the "What-If" scenario slider
2. **Overview** → Check the revenue pie chart
3. **Segmentation** → Explore the RFM scatter plot
4. **Churn Predictions** → Download the high-risk customer CSV

---

## Questions?

If you encounter issues, please open a GitHub issue or contact me directly.
