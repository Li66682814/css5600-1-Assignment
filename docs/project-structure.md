# Project structure

```text
css5600-1-Assignment/
├── README.md
├── README_ZH.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── Assignment.docx
│   ├── project-structure.md
│   └── project-structure_ZH.md
├── Question1/
│   ├── question1.py
│   └── Question1.csv
└── Question2/
    ├── question2.py
    └── Question2Data/
        ├── DecisionTreeParameterGrid.csv
        ├── telco-customer-churn.zip
        └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

| File | Purpose |
| --- | --- |
| `Assignment.docx` | Assignment report containing the Question 1 calculations and the Question 2 model evaluation and discussion |
| `question1.py` | Naive Bayes and Gaussian Naive Bayes experiment for Question 1 |
| `Question1.csv` | Input data for Question 1 |
| `question2.py` | Data preparation, prediction, and evaluation for Question 2 |
| `DecisionTreeParameterGrid.csv` | Candidate hyperparameter combinations for the decision tree |
| `telco-customer-churn.zip` | Compressed dataset for Question 2 |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Extracted dataset for Question 2 |

## Question 2 data

Question 2 uses the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).  
The script reads the local CSV if available. Otherwise, it tries to extract the local ZIP archive.  
If both files are missing, the script downloads the ZIP using `curl`.
