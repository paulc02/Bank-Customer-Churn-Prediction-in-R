# Bank Customer Churn Prediction in R

> Predicting which bank customers are about to leave, using **fuzzy clustering, logistic regression, decision trees, KNN and a neural network**, all compared on the same 10,000-customer dataset.

🇷🇴 [Versiune în română](README.ro.md) · 📄 [Full report (PDF, Romanian)](docs/Proiect_ICE_report_RO.pdf)

---

## Overview

Customer churn is one of the costliest problems for banks: keeping a customer is much cheaper than acquiring a new one. This project analyses the behaviour of customers of a European bank (France, Germany, Spain) and answers three questions:

1. **Are there natural customer segments?** → fuzzy clustering (FANNY)
2. **Which factors drive churn?** → binomial logistic regression and decision trees
3. **Which model predicts churn best?** → logistic regression vs. decision trees vs. KNN vs. neural network

It was developed as a university project for the *Computational Intelligence* course at the Bucharest University of Economic Studies (ASE), Faculty of Cybernetics, Statistics and Economic Informatics.

## Key results

| Model                        | Accuracy | AUC        |
| ---------------------------- | -------- | ---------- |
| Logistic regression          | 81.12 %  | 0.7656     |
| Decision tree (initial)      | 86.10 %  | 0.7675     |
| Decision tree (pruned)       | 85.50 %  | 0.7666     |
| KNN (k = 13)                 | 83.63 %  | 0.8072     |
| **Neural network (11-5-3-1)** | 86.00 %  | **0.8637** |

![Model comparison](images/03_model_comparison.png)

**Main findings**

- The **neural network** has the best discriminative power (AUC 0.864), followed by KNN (0.807).
- **Age** is the strongest risk factor: customers over ~43 churn at about 41 % vs. 12 % for younger ones (first split of the decision tree).
- **Active members** are far less likely to leave; **German customers** churn about twice as often as French or Spanish ones (~32 % vs. ~16–17 %); **women** churn more than men (~25 % vs. ~16 %).
- **Fuzzy clustering** finds two weakly separated segments: customers with high balances and more engagement vs. customers with almost empty accounts.
- Because only ~20 % of customers churn, **accuracy alone is misleading**, so AUC is the main comparison metric.

| Churn by age | Churn by country |
| :---: | :---: |
| ![Age vs churn](images/01_age_vs_churn.png) | ![Churn by country](images/02_churn_by_country.png) |

| Neural network ROC | Neural network architecture |
| :---: | :---: |
| ![NN ROC](images/04_neural_network_roc.png) | ![NN architecture](images/05_neural_network_architecture.png) |

## Dataset

[Bank Customer Churn](https://www.kaggle.com/) dataset (public, from Kaggle): 10,000 customers, 14 raw columns. `RowNumber`, `CustomerId` and `Surname` are dropped (no predictive value), leaving 10 predictors and the target:

| Variable | Type | Description |
| --- | --- | --- |
| CreditScore | numeric | Credit score (350–850) |
| Geography | categorical | France, Germany, Spain |
| Gender | categorical | Male, Female |
| Age | numeric | Age in years |
| Tenure | numeric | Years as a customer (0–10) |
| Balance | numeric | Account balance |
| NumOfProducts | numeric | Number of bank products held |
| HasCrCard | binary | Has a credit card |
| IsActiveMember | binary | Is an active member |
| EstimatedSalary | numeric | Estimated salary |
| **Exited** | binary | **Target:** 1 = left the bank (20.4 %), 0 = stayed (79.6 %) |

## Methods

| # | Technique | R packages |
| - | --- | --- |
| 1 | Descriptive statistics and EDA | `psych`, `ggplot2`, `corrplot` |
| 2 | Fuzzy clustering (FANNY), k chosen with Dunn coefficient and silhouette | `cluster`, `factoextra` |
| 3 | Binomial logistic regression (churn) and multinomial logistic regression (country) | `stats`, `nnet`, `caTools`, `pROC` |
| 4 | Classification and regression trees with pruning | `rpart`, `rpart.plot` |
| 5 | K-nearest neighbours with z-score normalisation | `class`, `kknn` |
| 6 | Neural network, 11-5-3-1, sigmoid, min-max scaling | `neuralnet`, `e1071` |

## Repository structure

```
bank-churn-prediction-r/
├── R/
│   ├── churn_analysis.R      # full analysis, chapter by chapter
│   └── install_packages.R    # installs the required packages
├── data/
│   └── bank_churn.csv        # dataset (Kaggle)
├── docs/
│   └── Proiect_ICE_report_RO.pdf   # full written report (Romanian)
├── images/                   # figures used in this README
├── bank-churn-prediction.Rproj
├── LICENSE
└── README.md
```

## How to run

1. Install [R](https://cran.r-project.org/) and [RStudio](https://posit.co/download/rstudio-desktop/).
2. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/bank-churn-prediction-r.git
   ```
3. Open `bank-churn-prediction.Rproj` in RStudio (this sets the working directory to the repo root).
4. Install the packages once: `source("R/install_packages.R")`
5. Run `R/churn_analysis.R`, either all at once or chapter by chapter.

All random steps use `set.seed(123)`, so results are reproducible on the same package versions.

## Limitations and next steps

- **Class imbalance:** sensitivity is low for most models (e.g. 0.22 for logistic regression, 0.32 for KNN) because the models favour the majority class. Next step: class weights, SMOTE or threshold tuning.
- **Single train/test split:** results come from one split. Cross-validation would give more reliable estimates. For KNN, `k` was chosen on the test set, which slightly favours its reported accuracy.
- **Weak cluster structure:** the average silhouette width is ~0.17, so the two segments overlap heavily and should be read as tendencies, not sharp groups.
- **Data:** richer behavioural features (transaction frequency, app usage, complaints) would likely improve every model.

## Author

**Calamusa Salvatore-Paolo**, Cybernetics and Economic Informatics, ASE Bucharest

## License

Released under the [MIT License](LICENSE). The dataset belongs to its original publisher on Kaggle and is subject to its own terms.
