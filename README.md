# Lightweight Agentic Framework for Interpretable ECG-Based MI Prediction

This repository contains a reproducible pipeline for predicting Myocardial Infarction (MI) from 12-lead ECGs using the PTB-XL dataset, augmented with an interpretable agentic workflow.

**⚠️ DISCLAIMER:** This system is a research prototype. It relies solely on ECG signals without clinical biomarkers (like troponin), patient history, or symptom data. The generated explanations and risk categories are meant for decision-support prototyping and **do not constitute a clinical diagnosis**. It has not been prospectively validated by clinicians.

## Features
- **Reproducible Pipeline:** Automated downloading of PTB-XL, standardized splitting, and transparent modeling.
- **Lightweight CNN:** A resource-efficient 1D CNN that can run on Colab CPU/GPU.
- **Agentic Explanation Workflow:** Converts model probabilities into structured, readable risk interpretations.

## Directory Structure
- `src/`: Contains source code for data loading, modeling, training, evaluation, and the agentic workflow.
- `notebooks/`: Contains `01_ECG_MI_Classification_PTBXL.ipynb` for easy Colab execution.
- `results/`: Contains generated metrics, prediction logs, plots, and agentic error analysis outputs.
- `paper/`: Contains the Overleaf-compatible LaTeX paper summarizing the work.

## Setup and Reproducibility

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
You can run the entire pipeline from end to end using `main.py`. Note that this will stream or download PTB-XL records.
```bash
python main.py
```
For a quick test on a subset of the data:
```bash
python main.py --max_samples 500 --epochs 5
```

### 3. Run via Colab Notebook
Open `notebooks/01_ECG_MI_Classification_PTBXL.ipynb` in Google Colab and run all cells.

## Results
After running, the `results/` directory will be populated with:
- `metrics.json`: AUROC, AUPRC, Accuracy, etc.
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`
- `error_analysis.csv` and `explanation_examples.json`: Containing the structured decision-support outputs.
