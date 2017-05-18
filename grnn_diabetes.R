pkgs <- c('caret', 'doParallel', 'foreach', 'grnn')
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
predict_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=grnn::guess(nn, as.matrix(i)))
  }
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(sig=seq(0.2, 1, 0.05), .combine=rbind) %dopar% {
  set.seed(101)
  model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=column), sigma=sig)
  model_grnn.pred <- pred_grnn(test_all[, -column], model_grnn)
  sse <- sum((test_all[, column] - model_grnn.pred)^2)
  data.frame(sig, sse=sse)
}

cat("\n### BEST SIGMA WITH THE LOWEST SSE ###\n")
print(best.sig <- cv[cv$sse==min(cv$sse), 1])

### BEST SIGMA WITH THE LOWEST SSE ###
# [1] 0.85
# > cv
# sig      sse
# 1  0.20 78.60869
# 2  0.25 75.65809
# 3  0.30 72.08739
# 4  0.35 68.11625
# 5  0.40 63.99199
# 6  0.45 60.03216
# 7  0.50 56.60772
# 8  0.55 53.89958
# 9  0.60 51.85717
# 10 0.65 50.34688
# 11 0.70 49.26007
# 12 0.75 48.53193
# 13 0.80 48.11732
# 14 0.85 47.96788
# 15 0.90 48.02806
# 16 0.95 48.24310
# 17 1.00 48.56685

set.seed(101)
model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=column), sigma=0.85)
model_grnn.pred <- predict_grnn(test_all[, -column], model_grnn)
model_grnn.predClass <- as.integer(round(model_grnn.pred))
test_class <- as.integer(test_all$V9)
u_grnn <- union(test_class, model_grnn.predClass)
caret::confusionMatrix(table(true=factor(test_class, u_grnn), predictions=factor(model_grnn.predClass, u_grnn)))

# Confusion Matrix and Statistics
# 
# predictions
# true   1   0
#     1  42  63
#     0  13 138
# 
# Accuracy : 0.7031          
# 95% CI : (0.6431, 0.7584)
# No Information Rate : 0.7852          
# P-Value [Acc > NIR] : 0.9992          
# 
# Kappa : 0.3385          
# Mcnemar's Test P-Value : 1.902e-08       
#                                           
#             Sensitivity : 0.7636          
#             Specificity : 0.6866          
#          Pos Pred Value : 0.4000          
#          Neg Pred Value : 0.9139          
#              Prevalence : 0.2148          
#          Detection Rate : 0.1641          
#    Detection Prevalence : 0.4102          
#       Balanced Accuracy : 0.7251          
#                                           
#        'Positive' Class : 1 
