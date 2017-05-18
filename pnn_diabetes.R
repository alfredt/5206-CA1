pkgs <- c('caret', 'doParallel', 'foreach', 'pnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_all <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")

# PRE-PROCESSING DATA
column = ncol(data_all)
data_all <- data.frame(scale(data_all[-ncol(data_all)]), data_all[ncol(data_all)])

require("caTools")
set.seed(101) 
sample = sample.split(data_all, SplitRatio = .75)
train_all = subset(data_all, sample == TRUE)
test_all  = subset(data_all, sample == FALSE)

# DEFINE A FUNCTION TO SCORE GRNN
predict_pnn <- function(x, pnn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=pnn::guess(pnn, as.matrix(i))$category)
  }
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(sigma=seq(0.2, 1, 0.05), .combine=rbind) %dopar% {
  set.seed(101)
  model_pnn <- pnn::smooth(pnn::learn(train_all, category.column=column), sigma=sigma)
  model_pnn.pred <- predict_pnn(test_all[, -column], model_pnn)
  u_pnn <- union(test_all[column], model_pnn.pred)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_pnn), predictions=factor(model_pnn.pred, u_pnn)))
  a <- m$overall[1]
  data.frame(sigma, accuracy=a)
}

cat("\n### BEST SIGMA WITH THE HIGHEST ACCURACY ###\n")
print(best.sigma <- cv[cv$accuracy==max(cv$accuracy), 1:2])

# sigma accuracy
# Accuracy11  0.75 0.765625
# 
# sigma  accuracy
# Accuracy    0.20 0.6718750
# Accuracy1   0.25 0.6718750
# Accuracy2   0.30 0.6757812
# Accuracy3   0.35 0.6796875
# Accuracy4   0.40 0.6835938
# Accuracy5   0.45 0.7070312
# Accuracy6   0.50 0.7265625
# Accuracy7   0.55 0.7265625
# Accuracy8   0.60 0.7304688
# Accuracy9   0.65 0.7304688
# Accuracy10  0.70 0.7421875
# Accuracy11  0.75 0.7656250
# Accuracy12  0.80 0.7578125
# Accuracy13  0.85 0.7617188
# Accuracy14  0.90 0.7578125
# Accuracy15  0.95 0.7578125
# Accuracy16  1.00 0.7539062

set.seed(101)
model_pnn <- pnn::smooth(pnn::learn(train_all, category.column=column), sigma=0.75)
model_pnn.predClass <- as.integer(predict_pnn(test_all[, -column], model_pnn))
test_class <- as.integer(test_all$V9)
u_pnn <- union(test_class, model_pnn.predClass)
caret::confusionMatrix(table(true=factor(test_class, u_pnn), predictions=factor(model_pnn.predClass, u_pnn)))
# 
# Confusion Matrix and Statistics
# 
# predictions
# true   1   0
#     1  73  32
#     0  28 123
# 
# Accuracy : 0.7656          
# 95% CI : (0.7089, 0.8161)
# No Information Rate : 0.6055          
# P-Value [Acc > NIR] : 4.257e-08       
# 
# Kappa : 0.5128          
# Mcnemar's Test P-Value : 0.6985          
#                                           
#             Sensitivity : 0.7228          
#             Specificity : 0.7935          
#          Pos Pred Value : 0.6952          
#          Neg Pred Value : 0.8146          
#              Prevalence : 0.3945          
#          Detection Rate : 0.2852          
#    Detection Prevalence : 0.4102          
#       Balanced Accuracy : 0.7582          
#                                           
#        'Positive' Class : 1