# CSS5600 Assignment 1

Week 4 机器学习作业包含两个实验。Question 1 用朴素贝叶斯和高斯朴素贝叶斯预测是否外出。Question 2 用决策树预测电信客户是否流失，并评估模型与主要分裂特征。

## 项目结构

```text
css5600-1-Assignment/
├── README.md
├── requirements.txt
├── .gitignore
├── Assignment.docx
├── Question1/
│   ├── question1.py
│   └── Question1.csv
└── Question2/
    ├── question2.py
    └── Question2Data/
        └── DecisionTreeParameterGrid.csv
```

| 文件 | 用途 |
| --- | --- |
| `Assignment.docx` | 作业报告，包含 Q1 手算和模型比较，以及 Q2 评估和讨论 |
| `question1.py` 与 `Question1.csv` | 读取 Q1 数据，复算条件概率和后验概率，并与 `GaussianNB` 比较 |
| `question2.py` | 准备 Telco Customer Churn 数据，训练并评估 `DecisionTreeClassifier` |
| `DecisionTreeParameterGrid.csv` | 保存 1,078 组候选超参数。每行是一组完整配置 |

## 安装依赖

在 Miniconda Prompt 中进入 `css5600-1-Assignment`，使用 Python 3.12。已有 `css5600-1` 环境时，从 `conda activate` 开始。

```bash
conda create -n css5600-1 python=3.12 pip
conda activate css5600-1
python -m pip install -r requirements.txt
```

## 运行作业

在项目根目录执行以下命令。

```bash
python Question1/question1.py
python Question2/question2.py
```

Q2 优先读取 `Question2Data` 中的本地 CSV，也可以使用同目录下的 ZIP 解压。如果两者都没有，脚本会调用 `curl` 从 Kaggle 下载数据，因此首次下载需要联网，并且 `curl` 需在 `PATH` 中。下载的 CSV 和 ZIP 是本地缓存，参数网格 CSV 随项目提交。

Q2 采用 80/20 分层划分，在训练集上对参数网格进行并行的 5 折交叉验证。用最佳参数在完整训练集上重训后，脚本会打印测试集准确率、混淆矩阵和特征重要性。
