pkgs <- c('nnet', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_nnet <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")
data_nnet$quality <- as.factor(data_nnet$quality)

# PRE-PROCESSING DATA
column = ncol(data_nnet)
data_nnet <- data.frame(scale(data_nnet[-column]), data_nnet[column])

require("caTools")
set.seed(101) 
sample = sample.split(data_nnet, SplitRatio = .75)
train_nnet = subset(data_nnet, sample == TRUE)
test_nnet  = subset(data_nnet, sample == FALSE)

#######################################
## nnet models
#######################################
# set.seed(101) 
# model_nnet <- nnet(train_nnet$quality ~ ., data=train_nnet, size=8, maxit=1000)
# model_nnet.pred <- predict(model_nnet, test_nnet)
# model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
# u_nnet <- union(test_nnet$quality, model_nnet.predClass)
# m <- caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(2, 20, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_nnet <- nnet::nnet(train_nnet$quality ~ ., data=train_nnet, size=node, maxit=1000)
  model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
  u_nnet <- union(test_nnet$quality, model_nnet.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])

### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###
# node  accuracy
# Accuracy3    5 0.5596405
# Accuracy6    8 0.5596405
# > cv
# node  accuracy
# Accuracy      2 0.5187908
# Accuracy1     3 0.5424837
# Accuracy2     4 0.5359477
# Accuracy3     5 0.5596405
# Accuracy4     6 0.5424837
# Accuracy5     7 0.5367647
# Accuracy6     8 0.5596405
# Accuracy7     9 0.5555556
# Accuracy8    10 0.5351307
# Accuracy9    11 0.5294118
# Accuracy10   12 0.5310458
# Accuracy11   13 0.5449346
# Accuracy12   14 0.5457516
# Accuracy13   15 0.5514706
# Accuracy14   16 0.5359477
# Accuracy15   17 0.5571895
# Accuracy16   18 0.5228758
# Accuracy17   19 0.5089869
# Accuracy18   20 0.5359477

set.seed(101)
model_nnet <- nnet::nnet(train_nnet$quality ~ ., data=train_nnet, size=8, maxit=1000)
model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
u_nnet <- union(test_nnet$quality, model_nnet.predClass)
caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))

# Confusion Matrix and Statistics
# 
# predictions
# true   6   5   8   4   7   3   9
#     6 396  92   4   2  47   0   1
#     5 125 212   0   4   4   0   0
#     8  35   0   3   0  14   0   0
#     4  16  28   0   2   1   0   0
#     7 147   8   5   0  72   0   1
#     3   2   2   0   0   0   0   0
#     9   1   0   0   0   0   0   0
# 
# Overall Statistics
# 
# Accuracy : 0.5596          
# 95% CI : (0.5313, 0.5877)
# No Information Rate : 0.5899          
# P-Value [Acc > NIR] : 0.9851          
# 
# Kappa : 0.3097          
# Mcnemar's Test P-Value : NA              
# 
# Statistics by Class:
# 
#                      Class: 6 Class: 5 Class: 8 Class: 4 Class: 7 Class: 3 Class: 9
# Sensitivity            0.5485   0.6199 0.250000 0.250000  0.52174       NA 0.000000
# Specificity            0.7092   0.8492 0.959571 0.962993  0.85175 0.996732 0.999182
# Pos Pred Value         0.7306   0.6145 0.057692 0.042553  0.30901       NA 0.000000
# Neg Pred Value         0.5220   0.8521 0.992321 0.994902  0.93340       NA 0.998365
# Prevalence             0.5899   0.2794 0.009804 0.006536  0.11275 0.000000 0.001634
# Detection Rate         0.3235   0.1732 0.002451 0.001634  0.05882 0.000000 0.000000
# Detection Prevalence   0.4428   0.2819 0.042484 0.038399  0.19036 0.003268 0.000817
# Balanced Accuracy      0.6288   0.7345 0.604785 0.606497  0.68674       NA 0.499591