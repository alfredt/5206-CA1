pkgs <- c('nnet', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_nnet <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")
data_nnet$V9 <- as.factor(data_nnet$V9)

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
  model_nnet <- nnet::nnet(train_nnet$V9 ~ ., data=train_nnet, size=node, maxit=1000)
  model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
  u_nnet <- union(test_nnet$quality, model_nnet.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_nnet$V9, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])

### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###
# node  accuracy
# Accuracy1    3 0.7851562
# > cv
# node  accuracy
# Accuracy      2 0.7382812
# Accuracy1     3 0.7851562
# Accuracy2     4 0.7421875
# Accuracy3     5 0.6953125
# Accuracy4     6 0.7148438
# Accuracy5     7 0.7070312
# Accuracy6     8 0.6953125
# Accuracy7     9 0.7109375
# Accuracy8    10 0.6640625
# Accuracy9    11 0.7070312
# Accuracy10   12 0.6562500
# Accuracy11   13 0.6757812
# Accuracy12   14 0.7109375
# Accuracy13   15 0.6835938
# Accuracy14   16 0.6679688
# Accuracy15   17 0.6757812
# Accuracy16   18 0.6601562
# Accuracy17   19 0.6718750
# Accuracy18   20 0.6914062

set.seed(101)
model_nnet <- nnet::nnet(train_nnet$V9 ~ ., data=train_nnet, size=3, maxit=1000)
model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
u_nnet <- union(test_nnet$quality, model_nnet.predClass)
caret::confusionMatrix(table(true=factor(test_nnet$V9, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))
# 
# Confusion Matrix and Statistics
# 
# predictions
# true   1   0
#     1  66  39
#     0  16 135
# 
# Accuracy : 0.7852          
# 95% CI : (0.7297, 0.8339)
# No Information Rate : 0.6797          
# P-Value [Acc > NIR] : 0.0001227       
# 
# Kappa : 0.5406          
# Mcnemar's Test P-Value : 0.0030123       
#                                           
#             Sensitivity : 0.8049          
#             Specificity : 0.7759          
#          Pos Pred Value : 0.6286          
#          Neg Pred Value : 0.8940          
#              Prevalence : 0.3203          
#          Detection Rate : 0.2578          
#    Detection Prevalence : 0.4102          
#       Balanced Accuracy : 0.7904          
#                                           
#        'Positive' Class : 1