# CSS5600 Assignment 1

English | [中文](README_ZH.md)

![UPM](https://img.shields.io/badge/University-UPM-9B1B30)
![Advanced Machine Learning](https://img.shields.io/badge/Course-Advanced%20Machine%20Learning-1D4ED8)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)

This Advanced Machine Learning assignment covers Weeks 4 and 5 and contains two experiments.

- Question 1 uses Naive Bayes and Gaussian Naive Bayes to classify the given sample.
- Question 2 uses a decision tree to predict telecom customer churn, evaluate its performance, and identify the features most important to its splits.

## Project structure

The assignment report is available as a [PDF](docs/Assignment.pdf) and a [Word document](docs/Assignment.docx).

See the [project structure guide](docs/project-structure.md) for the directory tree, file descriptions, and data source.

## Install dependencies

The project uses Python 3.12.

```bash
conda create -n your_env_name python=3.12 pip
conda activate your_env_name
python -m pip install -r requirements.txt
```

## Run the experiments

Run these commands from the repository root.

```bash
python Question1/question1.py
python Question2/question2.py
```
