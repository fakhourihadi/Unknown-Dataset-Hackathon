# Unknown Dataset Hackathon

**EECE 490 | Introduction to Machine Learning**

A machine learning project designed to process unfamiliar tabular datasets using a single, reusable Python program. The pipeline prepares the data, trains a suitable model, generates predictions, and produces a structured dataset profile.

## Project Overview

The hackathon focuses on three machine learning tasks:

| Task | Goal | Algorithms |
| --- | --- | --- |
| **Classification** | Predict categorical labels | Extra Trees, Random Forest |
| **Regression** | Predict numerical values | Extra Trees, Random Forest |
| **Clustering** | Group similar data points | K-Means, MiniBatch K-Means |

The pipeline handles common data challenges, including missing values, categorical variables, constant or duplicate columns, and features with different scales.

## How It Works

1. **Data inspection:** Analyze the input dataset and identify its features and quality issues.
2. **Preprocessing:** Handle missing data, remove unsuitable columns, and encode categorical features.
3. **Modeling:** Compare candidate models for supervised tasks or apply clustering with a specified number of groups.
4. **Output:** Save predictions and a structured dataset profile.

## Repository Structure

```text
Unknown-Dataset-Hackathon/
├── main.py             # Main machine learning pipeline
├── requirements.txt    # Python dependencies
├── smoke_test.py       # Basic functionality tests
└── README.md           # Project documentation
```

## Installation

Python 3.11 is recommended. After downloading or cloning this repository, open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## Running the Project

Specify the task and the input CSV file(s). The examples below show the program's command-line interface.

**Classification**

```bash
python main.py --train train.csv --test test.csv --task classification --target Churn
```

**Regression**

```bash
python main.py --train train.csv --test test.csv --task regression --target SalePrice
```

**Clustering**

```bash
python main.py --train customers.csv --task clustering --clusters 5
```

The CSV names above are examples; these datasets are not included in the repository. For classification and regression, the training file must contain the target column.

## Outputs

- **`predictions.csv`** — Predicted classes, numerical values, or cluster assignments.
- **`profile.json`** — Dataset characteristics, selected model information, and execution details.

## Testing

Run the included basic functionality tests with:

```bash
python smoke_test.py
```

These tests check whether all three task types execute; they do not measure performance on the hackathon's private datasets.

## Team Members

- Hadi Fakhouri
- Elie Chakhtoura
- Malek El Wattar

## Important Note

The current command-line interface and output format have not been verified against the hackathon's official checker. The program also does not guarantee completion within the 60-second assessment limit.
