# Run once to install every package used by churn_analysis.R
pkgs <- c("psych", "ggplot2", "corrplot", "e1071", "cluster", "factoextra",
          "caTools", "nnet", "pROC", "rpart", "rpart.plot", "class", "kknn",
          "neuralnet")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
