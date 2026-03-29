# Toxic Comment Classification Analysis

A machine learning analysis of online comment toxicity using logistic regression 
and TF-IDF text representation, built to explore content moderation challenges 
at scale.

## Overview

This project builds and evaluates a binary classifier to detect toxic comments 
using the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge), 
originally released by Jigsaw/Google in 2018. The dataset contains ~160,000 
Wikipedia talk page comments labelled across six toxicity categories.

## Key Findings

- Final model achieved **AUC of 0.967**, demonstrating strong discriminative ability
- At a classification threshold of 0.6: **precision 0.72, recall 0.81, F1 0.76**
- Toxic comments tend to be significantly shorter (~23 words median) than clean 
comments (~38 words)
- Class imbalance (90% clean, 10% toxic) was addressed using `class_weight='balanced'`
- Logistic Regression outperformed Multinomial Naive Bayes baseline on recall

## Project Structure
```
├── notebook.ipynb       # Full analysis notebook
├── README.md            # Project overview
└── data/                # Not included — download from Kaggle (see below)
```

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/dylan-130/Comment-Toxicity-Analysis.git
cd toxic-comment-analysis
```

**2. Download the dataset**

Download `train.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) and place it in the root directory.

**3. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**4. Run the notebook**
```bash
jupyter notebook notebook.ipynb
```

## Methodology

- **Text preprocessing** — lowercasing, regex cleaning, whitespace normalisation
- **Feature engineering** — TF-IDF with unigrams and bigrams (20,000 features)
- **Modelling** — Logistic Regression with L2 regularisation and class weighting
- **Evaluation** — precision, recall, F1, ROC-AUC, confusion matrix
- **Threshold optimisation** — explored thresholds from 0.2 to 0.6 to balance 
precision and recall

## Libraries

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn

## Author

Dylan Byrne | [LinkedIn](https://www.linkedin.com/in/dylan-byrne01/) | [GitHub](https://github.com/dylan-130)