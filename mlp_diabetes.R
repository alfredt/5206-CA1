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
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_all[column], size=node, maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- round(model_mlp.pred)
  u_mlp <- union(test_all[column], model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])
# node  accuracy
# Accuracy    2 0.7686275
# 
# node  accuracy
# Accuracy      2 0.7686275
# Accuracy1     3 0.7254902
# Accuracy2     4 0.7333333
# Accuracy3     5 0.7529412
# Accuracy4     6 0.7372549
# Accuracy5     7 0.7411765
# Accuracy6     8 0.7176471
# Accuracy7     9 0.7098039
# Accuracy8    10 0.7098039
# Accuracy9    11 0.7254902
# Accuracy10   12 0.7019608
# Accuracy11   13 0.7411765
# Accuracy12   14 0.6901961
# Accuracy13   15 0.7137255
# Accuracy14   16 0.6941176
# Accuracy15   17 0.7215686
# Accuracy16   18 0.7137255
# Accuracy17   19 0.6901961
# Accuracy18   20 0.7333333

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv2 <- foreach(node=seq(2, 10, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_all[column], size=c(2, node), maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- round(model_mlp.pred)
  u_mlp <- union(test_all[column], model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node2 <- cv2[cv2$accuracy==max(cv2$accuracy), 1:2])
# node  accuracy
# Accuracy2    4 0.7764706
# Accuracy6    8 0.7764706
# 
# node  accuracy
# Accuracy     2 0.7568627
# Accuracy1    3 0.7215686
# Accuracy2    4 0.7764706
# Accuracy3    5 0.7411765
# Accuracy4    6 0.7607843
# Accuracy5    7 0.7647059
# Accuracy6    8 0.7764706
# Accuracy7    9 0.7490196
# Accuracy8   10 0.7529412

set.seed(101)
model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_all[column], size=c(2,4), maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[-column])
model_mlp.predClass <- as.integer(round(model_mlp.pred))
test_class <- as.integer(test_all$X1)
u_mlp <- union(test_class, model_mlp.predClass)
caret::confusionMatrix(table(true=factor(test_class, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
# 
# Confusion Matrix and Statistics
# 
# predictions
# true   0   1
#     0 141  23
#     1  34  57
# 
# Accuracy : 0.7765          
# 95% CI : (0.7203, 0.8261)
# No Information Rate : 0.6863          
# P-Value [Acc > NIR] : 0.0009027       
# 
# Kappa : 0.4996          
# Mcnemar's Test P-Value : 0.1853263       
#                                           
#             Sensitivity : 0.8057          
#             Specificity : 0.7125          
#          Pos Pred Value : 0.8598          
#          Neg Pred Value : 0.6264          
#              Prevalence : 0.6863          
#          Detection Rate : 0.5529          
#    Detection Prevalence : 0.6431          
#       Balanced Accuracy : 0.7591          
#                                           
#        'Positive' Class : 0