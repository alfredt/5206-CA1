pkgs <- c('RSNNS', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_all <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")

# PRE-PROCESSING DATA
column = ncol(data_all)
data_all <- data.frame(scale(data_all[-ncol(data_all)]), data_all[ncol(data_all)])

require("caTools")
set.seed(101) 
sample = sample.split(data_all, SplitRatio = .75)
train_all = subset(data_all, sample == TRUE)
test_all  = subset(data_all, sample == FALSE)
train_target <- decodeClassLabels(unlist(train_all[column]))

predict_class <- function(pred) {
  pred <- data.frame(pred)
  names(pred) <- c("3", "4", "5", "6", "7", "8", "9")
  return(names(pred)[apply(pred, 1, which.max)]) 
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(2, 20, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::rbf(x=train_all[-column], y=train_target, size=node, maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- predict_class(model_mlp.pred)
  u_mlp <- union(test_all$quality, model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])

# node  accuracy
# Accuracy16   18 0.5343137
# Accuracy18   20 0.5343137
# 
# node  accuracy
# Accuracy      2 0.4452614
# Accuracy1     3 0.4517974
# Accuracy2     4 0.4534314
# Accuracy3     5 0.5024510
# Accuracy4     6 0.4950980
# Accuracy5     7 0.4910131
# Accuracy6     8 0.5130719
# Accuracy7     9 0.5147059
# Accuracy8    10 0.5130719
# Accuracy9    11 0.5302288
# Accuracy10   12 0.5220588
# Accuracy11   13 0.5204248
# Accuracy12   14 0.5269608
# Accuracy13   15 0.5277778
# Accuracy14   16 0.5253268
# Accuracy15   17 0.5294118
# Accuracy16   18 0.5343137
# Accuracy17   19 0.5285948
# Accuracy18   20 0.5343137

set.seed(101)
model_mlp <- RSNNS::rbf(x=train_all[-column], y=train_target, size=20, maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[-column])
model_mlp.predClass <- predict_class(model_mlp.pred)
u_mlp <- union(test_all$quality, model_mlp.predClass)
caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))

# Confusion Matrix and Statistics
# 
# predictions
# true   6   5   8   4   7   3   9
#     6 407 109   0   0  26   0   0
#     5 144 199   0   0   2   0   0
#     8  41   0   0   0  11   0   0
#     4  13  34   0   0   0   0   0
#     7 176   9   0   0  48   0   0
#     3   1   3   0   0   0   0   0
#     9   0   0   0   0   1   0   0
# 
# Overall Statistics
# 
# Accuracy : 0.5343          
# 95% CI : (0.5059, 0.5626)
# No Information Rate : 0.6389          
# P-Value [Acc > NIR] : 1               
# 
# Kappa : 0.2512          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 6 Class: 5 Class: 8 Class: 4 Class: 7 Class: 3 Class: 9
# Sensitivity            0.5205   0.5621       NA       NA  0.54545       NA       NA
# Specificity            0.6946   0.8322  0.95752   0.9616  0.83715 0.996732 0.999183
# Pos Pred Value         0.7509   0.5768       NA       NA  0.20601       NA       NA
# Neg Pred Value         0.4501   0.8237       NA       NA  0.95964       NA       NA
# Prevalence             0.6389   0.2892  0.00000   0.0000  0.07190 0.000000 0.000000
# Detection Rate         0.3325   0.1626  0.00000   0.0000  0.03922 0.000000 0.000000
# Detection Prevalence   0.4428   0.2819  0.04248   0.0384  0.19036 0.003268 0.000817
# Balanced Accuracy      0.6075   0.6972       NA       NA  0.69130       NA       NA