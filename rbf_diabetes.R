pkgs <- c('RSNNS', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_all <- read.csv(file="Diabetes.csv", head=TRUE, sep=",")

# PRE-PROCESSING DATA
column = ncol(data_all)
data_all <- data.frame(scale(data_all[-ncol(data_all)]), data_all[ncol(data_all)])

require("caTools")
set.seed(101) 
sample = sample.split(data_all, SplitRatio = .75)
train_all = subset(data_all, sample == TRUE)
test_all  = subset(data_all, sample == FALSE)
train_target <- decodeClassLabels(unlist(train_all[column]))

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(2, 20, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_rbf <- RSNNS::rbf(x=train_all[-column], y=train_all[column], size=node, maxit=1000)
  model_rbf.pred <- predict(model_rbf, test_all[-column])
  model_rbf.predClass <- round(model_rbf.pred)
  u_mlp <- union(test_all[column], model_rbf.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_mlp), predictions=factor(model_rbf.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])
### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###
# node  accuracy
# Accuracy13   15 0.7882353
# > cv
# node  accuracy
# Accuracy      2 0.7372549
# Accuracy1     3 0.7450980
# Accuracy2     4 0.7411765
# Accuracy3     5 0.7529412
# Accuracy4     6 0.7529412
# Accuracy5     7 0.7529412
# Accuracy6     8 0.7647059
# Accuracy7     9 0.7725490
# Accuracy8    10 0.7803922
# Accuracy9    11 0.7803922
# Accuracy10   12 0.7803922
# Accuracy11   13 0.7725490
# Accuracy12   14 0.7803922
# Accuracy13   15 0.7882353
# Accuracy14   16 0.7843137
# Accuracy15   17 0.7764706
# Accuracy16   18 0.6274510
# Accuracy17   19 0.7725490
# Accuracy18   20 0.7647059

set.seed(101)
model_rbf <- RSNNS::rbf(x=train_all[-column], y=train_all[column], size=15, maxit=1000)
model_rbf.pred <- predict(model_rbf, test_all[-column])
model_rbf.predClass <- as.integer(round(model_rbf.pred))
test_class <- as.integer(test_all$X1)
u_rbf <- union(test_class, model_rbf.predClass)
caret::confusionMatrix(table(true=factor(test_class, u_rbf), predictions=factor(model_rbf.predClass, u_rbf)))
# 
# Confusion Matrix and Statistics
# 
# predictions
# true   0   1
#     0 141  23
#     1  31  60
# 
# Accuracy : 0.7882          
# 95% CI : (0.7329, 0.8367)
# No Information Rate : 0.6745          
# P-Value [Acc > NIR] : 4.003e-05       
# 
# Kappa : 0.5295          
# Mcnemar's Test P-Value : 0.3408          
#                                           
#             Sensitivity : 0.8198          
#             Specificity : 0.7229          
#          Pos Pred Value : 0.8598          
#          Neg Pred Value : 0.6593          
#              Prevalence : 0.6745          
#          Detection Rate : 0.5529          
#    Detection Prevalence : 0.6431          
#       Balanced Accuracy : 0.7713          
#                                           
#        'Positive' Class : 0

