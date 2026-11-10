# 项目结构

```text
css5600-1-Assignment/
├── README.md
├── README_ZH.md
├── requirements.txt
├── .gitignore
├── docs/
│   ├── Assignment.docx
│   ├── project-structure.md
│   ├── project-structure_ZH.md
│   └── readme-header.svg
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

| 文件 | 用途 |
| --- | --- |
| `Assignment.docx` | 作业报告，包含 Question 1 的计算过程和 Question 2 的模型评估与讨论 |
| `question1.py` | Question 1 的朴素贝叶斯和高斯朴素贝叶斯实验脚本 |
| `Question1.csv` | Question 1 的实验数据 |
| `question2.py` | Question 2 的数据准备、预测和评估脚本 |
| `DecisionTreeParameterGrid.csv` | 决策树候选超参数 |
| `telco-customer-churn.zip` | Question 2 实验数据压缩包 |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Question 2 实验数据解压文件 |

## Question 2 数据

Question 2 使用 [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 数据集  
脚本优先读取本地 CSV，找不到时尝试解压本地 ZIP  
当两者都缺少时，脚本调用 `curl` 下载 ZIP
