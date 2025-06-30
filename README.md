# BIODD Project - VM Setup Instructions

## Overview
This repository contains the BIODD project for pancreatic cancer drug discovery. The code has been optimized to run on cloud VMs with limited memory.

## Setup Instructions

### 1. Upload and Extract Files
After uploading and extracting this zip file on your VM, you should have the following files:
- `1.Download_Bioactivity_Data.py`
- `2.Exploratory_Data_Analysis.py`
- `3.Descriptor_Calculation_and_Dataset_Preparation.py`
- `4.Regression_Models_with_Random_Forest.py`
- `5.Comparing_Regressors.py`
- `requirements.txt`
- Other data folders and support files

### 2. Create Virtual Environment
```bash
# Create and activate virtual environment (recommended)
python3 -m venv biodd_env
source biodd_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you have issues with RDKit installation, you can try:
```bash
pip install rdkit-pypi
```

### 4. Run the Pipeline

Run each script in sequence. Each script outputs files needed by the next script:

```bash
# Step 1: Download and preprocess bioactivity data
python3 1.Download_Bioactivity_Data.py

# Step 2: Perform exploratory data analysis
python3 2.Exploratory_Data_Analysis.py

# Step 3: Calculate molecular descriptors and prepare dataset
python3 3.Descriptor_Calculation_and_Dataset_Preparation.py

# Step 4: Create regression models with Random Forest
python3 4.Regression_Models_with_Random_Forest.py

# Step 5: Compare different regression models
python3 5.Comparing_Regressors.py
```

## Optimization Notes
- All scripts have been modified to run in single-core mode to prevent memory issues
- Progress reporting has been added to monitor long-running operations
- Parameter grids have been simplified to reduce memory usage
- You can monitor memory usage with `htop` or `free -m`

## Troubleshooting

### Memory Issues
If you encounter memory issues:
```bash
# Monitor memory usage
free -m
htop

# Increase swap space if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Process Management
```bash
# Run a script in background with output saved
nohup python3 5.Comparing_Regressors.py > regressor_output.log 2>&1 &

# Check on running processes
ps aux | grep python

# Check logs
tail -f regressor_output.log
```

## Expected Completion Times
- Steps 1-3: ~10-30 minutes each
- Step 4 (Random Forest): ~1-2 hours
- Step 5 (Multiple Regressors): ~3-6 hours

The modified scripts include progress indicators so you can track the status of long-running operations. 