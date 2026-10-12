# =============================================================================
# Bank Customer Churn Prediction - Computational Intelligence in R
# Author : Calamusa Salvatore-Paolo (ASE Bucuresti, CSIE - Cibernetica Economica)
# Run    : open bank-churn-prediction.Rproj in RStudio (so the working directory
#          is the repo root), run R/install_packages.R once, then run this script.
# Data   : data/bank_churn.csv  (Kaggle "Bank Customer Churn", 10,000 customers)
# =============================================================================

# CAPITOL 1: INSTALARE PACHETE NECESARE 

# install.packages(c("psych", "ggplot2", "corrplot", "e1071", "cluster",
#                    "factoextra", "caTools", "nnet", "pROC", "rpart",
#                    "rpart.plot", "class", "kknn", "neuralnet", "ROCR"))


 
# CAPITOL 2: DESCRIEREA DATELOR SI STATISTICI DESCRIPTIVE

# Incarcare date 
date <- read.csv("data/bank_churn.csv")

# Eliminam coloanele fara valoare predictiva
date <- date[, !names(date) %in% c("RowNumber", "CustomerId", "Surname")]

# Vizualizare structura
str(date)
head(date)

# Transformam variabilele categoriale in factori
date$Geography      <- as.factor(date$Geography)
date$Gender         <- as.factor(date$Gender)
date$HasCrCard      <- as.factor(date$HasCrCard)
date$IsActiveMember <- as.factor(date$IsActiveMember)
date$Exited         <- as.factor(date$Exited)

# Statistici descriptive pentru variabilele numerice
library(psych)
describe(date[, c("CreditScore", "Age", "Tenure", "Balance",
                  "NumOfProducts", "EstimatedSalary")])

# Distributia variabilei tinta
table(date$Exited)
prop.table(table(date$Exited))

# Distributia variabilelor categoriale
table(date$Geography)
table(date$Gender)

# Grafice descriptive
library(ggplot2)

# Histograma CreditScore
ggplot(date, aes(x = CreditScore, fill = Exited)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Distributia CreditScore pe categorii Exited",
       x = "Credit Score", y = "Frecventa") +
  scale_fill_manual(values = c("0" = "#2196F3", "1" = "#F44336"),
                    labels = c("Ramas", "Plecat")) +
  theme_minimal()

# Boxplot Age vs Exited
ggplot(date, aes(x = Exited, y = Age, fill = Exited)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Distributia varstei in functie de Exited",
       x = "Exited (0=Ramas, 1=Plecat)", y = "Varsta") +
  scale_fill_manual(values = c("0" = "#2196F3", "1" = "#F44336")) +
  theme_minimal()

# Rata de abandon pe tara
ggplot(date, aes(x = Geography, fill = Exited)) +
  geom_bar(position = "fill") +
  labs(title = "Rata de churn pe tara",
       x = "Tara", y = "Proportie") +
  scale_fill_manual(values = c("0" = "#2196F3", "1" = "#F44336"),
                    labels = c("Ramas", "Plecat")) +
  theme_minimal()

# Rata de abandon pe gen
ggplot(date, aes(x = Gender, fill = Exited)) +
  geom_bar(position = "fill") +
  labs(title = "Rata de churn pe gen",
       x = "Gen", y = "Proportie") +
  scale_fill_manual(values = c("0" = "#2196F3", "1" = "#F44336"),
                    labels = c("Ramas", "Plecat")) +
  theme_minimal()

# Matrice de corelatie pentru variabilele numerice
library(corrplot)
date_num <- date[, c("CreditScore", "Age", "Tenure", "Balance",
                     "NumOfProducts", "EstimatedSalary")]
corrplot(cor(date_num), method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black",
         title = "Matricea de corelatie")



# CAPITOL 3: CLUSTERIZARE FUZZY

library(cluster)
library(factoextra)

# Pregatire date
date_cluster <- date[, c("CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary")]
date_cluster <- scale(date_cluster)

# Esantion 2000 observatii
set.seed(123)
idx_sample <- sample(1:nrow(date_cluster), 2000)
date_cluster_sample <- date_cluster[idx_sample, ]

# Determinare numar optim de clustere
k_values    <- 2:6
dunn_values <- numeric(length(k_values))
sil_values  <- numeric(length(k_values))

# Metoda 1: Coeficientul Dunn
for(i in 1:length(k_values)){
  res <- fanny(date_cluster_sample, k = k_values[i])
  dunn_values[i] <- res$coeff["dunn_coeff"]
}
data.frame(k = k_values, Dunn = dunn_values)
plot(k_values, dunn_values, type = "b", pch = 19,
     xlab = "Numar clustere", ylab = "Dunn coefficient",
     main = "Alegerea lui k (Dunn)")
k_opt_dunn <- k_values[which.max(dunn_values)]
k_opt_dunn

# Metoda 2: Silhouette
for(i in 1:length(k_values)){
  res <- fanny(date_cluster_sample, k = k_values[i])
  sil_values[i] <- mean(res$silinfo$widths[, 3])
}
data.frame(k = k_values, Silhouette = sil_values)
plot(k_values, sil_values, type = "b", pch = 19,
     xlab = "Numar clustere", ylab = "Silhouette",
     main = "Alegerea lui k (Silhouette)")
k_opt <- k_values[which.max(sil_values)]
k_opt

# Model FANNY final cu k optim 
set.seed(123)
res.fanny <- fanny(date_cluster_sample, k_opt)

data.frame(res.fanny$clustering)

fviz_cluster(res.fanny, ellipse.type = "convex", repel = FALSE,
             palette = "jco", ggtheme = theme_minimal(), legend = "right")

fviz_silhouette(res.fanny, palette = "jco", ggtheme = theme_minimal())

head(res.fanny$membership, 5)
res.fanny$silinfo

# Mediile pe clustere 
aggregate(date[idx_sample, c("CreditScore", "Age", "Tenure",
                             "Balance", "EstimatedSalary")],
          by = list(Cluster = res.fanny$clustering),
          FUN = mean)



# CAPITOL 4: REGRESIA LOGISTICA BINOMIALA SI MULTINOMIALA

library(caTools)

date$Exited_num <- as.numeric(as.character(date$Exited))

# Regresia logistica BINOMIALA 
# Variabila dependenta: Exited 
set.seed(123)
split <- sample.split(date$Exited_num, SplitRatio = 0.75)
setantrenare <- subset(date, split == TRUE)
settestare   <- subset(date, split == FALSE)

model_binom <- glm(Exited_num ~ CreditScore + Geography + Gender +
                     Age + Tenure + Balance + NumOfProducts +
                     HasCrCard + IsActiveMember + EstimatedSalary,
                   data = setantrenare,
                   family = binomial(link = "logit"))

summary(model_binom)

# Odds Ratios
exp(coef(model_binom))

# Predictii pe setul de testare
prob_test <- predict(model_binom, settestare, type = "response")
pred_class <- ifelse(prob_test > 0.5, 1, 0)

# Matricea de confuzie
conf_mat_binom <- table(Predicted = pred_class,
                        Actual = settestare$Exited_num)
print(conf_mat_binom)

# Acuratete
accuracy_binom <- mean(pred_class == settestare$Exited_num)
cat("Acuratete model binomial:", round(accuracy_binom * 100, 2), "%\n")

# Metrici suplimentare
TP <- conf_mat_binom["1", "1"]
TN <- conf_mat_binom["0", "0"]
FP <- conf_mat_binom["1", "0"]
FN <- conf_mat_binom["0", "1"]

sensitivitate <- TP / (TP + FN)
specificitate <- TN / (TN + FP)
cat("Senzitivitate:", round(sensitivitate, 3), "\n")
cat("Specificitate:", round(specificitate, 3), "\n")

# Curba ROC si AUC
library(pROC)
roc_binom <- roc(settestare$Exited_num, prob_test)
plot(roc_binom, col = "blue", lwd = 2,
     main = "Curba ROC - Regresie logistica binomiala")
abline(a = 0, b = 1, lty = 2, col = "gray")
auc_binom_val <- round(auc(roc_binom), 4)
cat("AUC:", auc_binom_val, "\n")

# Prag optim 
prag_optim <- as.numeric(coords(roc_binom, "best", ret = "threshold"))
pred_optim <- ifelse(prob_test > prag_optim, 1, 0)
conf_optim <- table(Predicted = pred_optim,
                    Actual = settestare$Exited_num)
print(conf_optim)
accuracy_binom_optim <- round(mean(pred_optim == settestare$Exited_num) * 100, 2)
cat("Acuratete cu prag optim:", accuracy_binom_optim, "%\n")


# Regresia logistica MULTINOMIALA 
# Variabila dependenta: Geography (France, Germany, Spain)

library(nnet)

date$Geography <- relevel(date$Geography, ref = "France")

set.seed(123)
split_m <- sample.split(date$Geography, SplitRatio = 0.75)
train_m <- subset(date, split_m == TRUE)
test_m  <- subset(date, split_m == FALSE)

model_multi <- multinom(Geography ~ CreditScore + Gender +
                          Age + Tenure + Balance + NumOfProducts +
                          HasCrCard + IsActiveMember + EstimatedSalary +
                          Exited_num,
                        data = train_m,
                        trace = FALSE)

summary(model_multi)

# Odds Ratios 
exp(coef(model_multi))

# Predictii
pred_multi  <- predict(model_multi, test_m)
conf_multi  <- table(Actual = test_m$Geography, Predicted = pred_multi)
print(conf_multi)

accuracy_multi <- mean(test_m$Geography == pred_multi)
cat("Acuratete model multinomial:", round(accuracy_multi * 100, 2), "%\n")

# Probabilitati de apartinere la fiecare clasa (primele 5 obs.)
head(predict(model_multi, test_m, type = "prob"), 5)



# CAPITOL 5: ARBORI DE REGRESIE SI CLASIFICARE

library(rpart)
library(rpart.plot)

# Pregatim datele 
date_tree <- date[, c("CreditScore", "Geography", "Gender", "Age",
                      "Tenure", "Balance", "NumOfProducts",
                      "HasCrCard", "IsActiveMember", "EstimatedSalary",
                      "Exited", "Exited_num")]

set.seed(123)
idx_tree     <- sample(1:nrow(date_tree), nrow(date_tree) * 0.8)
train_tree   <- date_tree[idx_tree, ]
test_tree    <- date_tree[-idx_tree, ]

# Arbore de CLASIFICARE 
arbore_cls <- rpart(Exited ~ CreditScore + Geography + Gender +
                      Age + Tenure + Balance + NumOfProducts +
                      HasCrCard + IsActiveMember + EstimatedSalary,
                    data = train_tree,
                    method = "class")

# Vizualizare arbore initial
rpart.plot(arbore_cls, extra = 104, cex = 0.7,
           main = "Arbore de clasificare - Churn bancar (inainte de pruning)")

print(arbore_cls)

# Evaluare initiala
pred_cls <- predict(arbore_cls, test_tree, type = "class")
conf_cls  <- table(Predicted = pred_cls, Actual = test_tree$Exited)
print(conf_cls)
accuracy_cls_init <- round(mean(pred_cls == test_tree$Exited) * 100, 2)
cat("Acuratete arbore initial:", accuracy_cls_init, "%\n")

# Curba ROC pentru arborele initial
prob_cls <- predict(arbore_cls, test_tree, type = "prob")
roc_cls  <- roc(as.numeric(as.character(test_tree$Exited)),
                prob_cls[, "1"])
auc_cls_init <- round(auc(roc_cls), 4)
cat("AUC arbore initial:", auc_cls_init, "\n")

# Pruning (curatarea arborelui) 
printcp(arbore_cls)

cp_optim <- arbore_cls$cptable[which.min(arbore_cls$cptable[, "xerror"]), "CP"]

arbore_cls_pruned <- prune(arbore_cls, cp = cp_optim)

rpart.plot(arbore_cls_pruned, extra = 104, cex = 0.7,
           main = "Arbore de clasificare - Churn bancar (dupa pruning)")

# Evaluare dupa pruning
pred_pruned <- predict(arbore_cls_pruned, test_tree, type = "class")
conf_pruned <- table(Predicted = pred_pruned, Actual = test_tree$Exited)
print(conf_pruned)
accuracy_cls_pruned <- round(mean(pred_pruned == test_tree$Exited) * 100, 2)
cat("Acuratete arbore pruned:", accuracy_cls_pruned, "%\n")

# Comparatie curbe ROC
prob_pruned <- predict(arbore_cls_pruned, test_tree, type = "prob")
roc_pruned  <- roc(as.numeric(as.character(test_tree$Exited)),
                   prob_pruned[, "1"])
auc_cls_pruned <- round(auc(roc_pruned), 4)

plot(roc_cls, col = "blue", lwd = 2,
     main = "ROC Curve - Arbore de clasificare (comparatie)")
lines(roc_pruned, col = "red", lwd = 2, lty = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
legend("bottomright",
       legend = c(paste("Initial (AUC =", auc_cls_init, ")"),
                  paste("Pruned (AUC =",  auc_cls_pruned, ")")),
       col = c("blue", "red"), lwd = 2, lty = c(1, 2))


# Arbore de REGRESIE (Balance ca variabila continua)
arbore_reg <- rpart(Balance ~ CreditScore + Geography + Gender +
                      Age + Tenure + NumOfProducts +
                      HasCrCard + IsActiveMember + EstimatedSalary,
                    data = train_tree,
                    method = "anova")

rpart.plot(arbore_reg, extra = 101, cex = 0.7,
           main = "Arbore de regresie - Sold cont bancar (inainte de pruning)")

# Predictii si RMSE initial
pred_reg  <- predict(arbore_reg, test_tree)
rmse_init <- sqrt(mean((test_tree$Balance - pred_reg)^2))
cat("RMSE arbore regresie initial:", round(rmse_init, 2), "\n")

# Pruning
printcp(arbore_reg)
cp_reg    <- arbore_reg$cptable[which.min(arbore_reg$cptable[, "xerror"]), "CP"]
arbore_reg_pruned <- prune(arbore_reg, cp = cp_reg)

rpart.plot(arbore_reg_pruned, extra = 101, cex = 0.7,
           main = "Arbore de regresie - Sold cont bancar (dupa pruning)")

pred_reg_pruned <- predict(arbore_reg_pruned, test_tree)
rmse_pruned_reg <- round(sqrt(mean((test_tree$Balance - pred_reg_pruned)^2)), 2)
cat("RMSE arbore regresie pruned:", rmse_pruned_reg, "\n")

# Comparatie valori reale vs previzionate
data.frame(Real = head(test_tree$Balance, 10),
           Previzionat = round(head(pred_reg_pruned, 10), 2))



# CAPITOL 6: KNN DE CLASIFICARE

library(class)
library(kknn)

# Pregatire date pentru KNN
df_knn <- date[, c("CreditScore", "Geography", "Gender", "Age",
                   "Tenure", "Balance", "NumOfProducts",
                   "HasCrCard", "IsActiveMember", "EstimatedSalary",
                   "Exited_num")]

# Creare variabile dummy pentru Geography si Gender
df_knn_matrix <- model.matrix(Exited_num ~ ., data = df_knn)[, -1]
df_knn_model  <- data.frame(df_knn_matrix, Exited = date$Exited_num)

# Normalizare cu scorul z
normalize <- function(x) (x - mean(x)) / sd(x)
df_knn_model[, -ncol(df_knn_model)] <-
  lapply(df_knn_model[, -ncol(df_knn_model)], normalize)
df_knn_model$Exited <- as.factor(df_knn_model$Exited)

# Impartire train/test 70/30
set.seed(123)
n_knn         <- nrow(df_knn_model)
train_idx_knn <- sample(1:n_knn, size = round(0.7 * n_knn))

train_knn    <- df_knn_model[train_idx_knn,  -ncol(df_knn_model)]
test_knn     <- df_knn_model[-train_idx_knn, -ncol(df_knn_model)]
train_labels <- df_knn_model$Exited[train_idx_knn]
test_labels  <- df_knn_model$Exited[-train_idx_knn]

# Alegerea valorii optime a lui k 
k_values   <- 1:20
accuracies <- numeric(length(k_values))

for (i in seq_along(k_values)) {
  pred_k        <- knn(train = train_knn, test = test_knn,
                       cl = train_labels, k = k_values[i])
  conf_k        <- table(pred_k, test_labels)
  accuracies[i] <- sum(diag(conf_k)) / sum(conf_k)
}

# Grafic acuratete vs k
plot(k_values, accuracies, type = "b", pch = 19,
     xlab = "Numarul de vecini (k)", ylab = "Acuratete",
     main = "KNN - Acuratete in functie de k")
best_k <- k_values[which.max(accuracies)]
abline(v = best_k, col = "red", lty = 2)
text(best_k, max(accuracies),
     labels = paste0("k optim = ", best_k),
     pos = 4, col = "red")
cat("Valoarea optima a lui k:", best_k, "\n")

# Model final cu k optim
pred_knn_final <- knn(train = train_knn, test = test_knn,
                      cl = train_labels, k = best_k)

conf_knn <- table(Predicted = pred_knn_final, Actual = test_labels)
print(conf_knn)

TP_knn <- conf_knn["1", "1"]
TN_knn <- conf_knn["0", "0"]
FP_knn <- conf_knn["1", "0"]
FN_knn <- conf_knn["0", "1"]

accuracy_knn    <- (TP_knn + TN_knn) / sum(conf_knn)
sensitivity_knn <- TP_knn / (TP_knn + FN_knn)
specificity_knn <- TN_knn / (TN_knn + FP_knn)
precision_knn   <- TP_knn / (TP_knn + FP_knn)

cat("\nEvaluare KNN cu k =", best_k, "\n")
cat("Acuratete    :", round(accuracy_knn,    3), "\n")
cat("Senzitivitate:", round(sensitivity_knn, 3), "\n")
cat("Specificitate:", round(specificity_knn, 3), "\n")
cat("Precizie     :", round(precision_knn,   3), "\n")

# Curba ROC pentru KNN
train_df_knn <- df_knn_model[train_idx_knn, ]
test_df_knn  <- df_knn_model[-train_idx_knn, ]

knn_prob_model <- kknn(Exited ~ ., train = train_df_knn,
                       test = test_df_knn, k = best_k,
                       kernel = "rectangular")

prob_knn_mat <- knn_prob_model$prob
prob_knn     <- prob_knn_mat[, "1"]  
true_knn     <- as.numeric(as.character(test_df_knn$Exited))

roc_knn <- roc(response = true_knn, predictor = prob_knn)
plot(roc_knn, col = "darkgreen", lwd = 2,
     main = paste("Curba ROC - KNN (k =", best_k, ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")
auc_knn_val <- round(auc(roc_knn), 4)
legend("bottomright",
       legend = paste("AUC =", auc_knn_val),
       col = "darkgreen", lwd = 2)
cat("AUC KNN:", auc_knn_val, "\n")



# CAPITOL 7: RETELE NEURONALE PENTRU CLASIFICARE
library(neuralnet)

# Pregatire date pentru reteaua neuronala
date_nn <- date[, c("CreditScore", "Age", "Tenure", "Balance",
                    "NumOfProducts", "EstimatedSalary", "Exited_num")]

# Adaugam variabile dummy pentru Geography si Gender
date_nn$Germany <- as.numeric(date$Geography == "Germany")
date_nn$Spain   <- as.numeric(date$Geography == "Spain")
date_nn$Male    <- as.numeric(date$Gender == "Male")
date_nn$HasCC   <- as.numeric(date$HasCrCard == 1)
date_nn$Active  <- as.numeric(date$IsActiveMember == 1)

# Normalizare min-max in [0,1]
maxim  <- apply(date_nn[, -7], 2, max)
minim  <- apply(date_nn[, -7], 2, min)
date_nn_scaled <- as.data.frame(scale(date_nn[, -7],
                                      center = minim,
                                      scale  = maxim - minim))
date_nn_scaled$Exited <- date_nn$Exited_num

# Impartire train/test (75%/25%)
set.seed(123)
library(caTools)
split_nn  <- sample.split(date_nn_scaled$Exited, SplitRatio = 0.75)
train_nn  <- subset(date_nn_scaled, split_nn == TRUE)
test_nn   <- subset(date_nn_scaled, split_nn == FALSE)

# Antrenarea retelei neuronale
set.seed(123)
retea_nn <- neuralnet(
  Exited ~ CreditScore + Age + Tenure + Balance +
    NumOfProducts + EstimatedSalary +
    Germany + Spain + Male + HasCC + Active,
  data          = train_nn,
  hidden        = c(5, 3),
  linear.output = FALSE,
  threshold     = 0.5,
  lifesign      = "minimal"
)

# Vizualizare arhitectura retea
plot(retea_nn, rep = "best")

# Predictii pe setul de testare
variabile_input <- c("CreditScore", "Age", "Tenure", "Balance",
                     "NumOfProducts", "EstimatedSalary",
                     "Germany", "Spain", "Male", "HasCC", "Active")
test_input   <- subset(test_nn, select = variabile_input)
rezultate_nn <- predict(retea_nn, test_input)

# Valori brute (probabilitati)
head(rezultate_nn)

# Clasificare binara cu prag 0.5
pred_nn <- as.numeric(rezultate_nn > 0.5)

# Matrice de confuzie
tab_nn <- table(Actual = test_nn$Exited, Predicted = pred_nn)
print(tab_nn)

# Metrici de performanta
library(e1071)
classAgreement(tab_nn)
accuracy_nn <- sum(diag(tab_nn)) / sum(tab_nn)
cat("Acuratete retea neuronala:", round(accuracy_nn * 100, 2), "%\n")

# Curba ROC si AUC 
library(pROC)
roc_nn <- roc(test_nn$Exited, as.numeric(rezultate_nn[, 1]))

plot(roc_nn, col = "purple", lwd = 2,
     main = "Curba ROC - Retea neuronala (arhitectura 11-5-3-1)")
abline(a = 0, b = 1, lty = 2, col = "gray")

auc_nn_val <- round(auc(roc_nn), 4)
cat("AUC retea neuronala:", auc_nn_val, "\n")
legend("bottomright",
       legend = paste("AUC =", auc_nn_val),
       col = "purple", lwd = 2)



# COMPARATIE FINALA A MODELELOR

# Colectam metricile calculate anterior in variabile
rezultate_finale <- data.frame(
  Model = c("Reg. Logistica Binomiala",
            "Arbore Clasificare (initial)",
            "Arbore Clasificare (pruned)",
            "KNN",
            "Retea Neuronala"),
  Acuratete_pct = c(
    round(accuracy_binom * 100, 2),
    accuracy_cls_init,
    accuracy_cls_pruned,
    round(accuracy_knn * 100, 2),
    round(accuracy_nn * 100, 2)
  ),
  AUC = c(
    auc_binom_val,
    auc_cls_init,
    auc_cls_pruned,
    auc_knn_val,
    auc_nn_val
  )
)

print(rezultate_finale)
