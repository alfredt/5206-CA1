pkgs <- c('caret', 'doParallel', 'foreach', 'grnn')
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

# DEFINE A FUNCTION TO SCORE GRNN
pred_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=guess(nn, as.matrix(i)))
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
# [1] 0.55
# 
# sig      sse
# 1  0.20 702.6699
# 2  0.25 669.2830
# 3  0.30 629.6584
# 4  0.35 588.3660
# 5  0.40 551.0654
# 6  0.45 522.5315
# 7  0.50 505.5045
# 8  0.55 500.3770
# 9  0.60 505.1758
# 10 0.65 516.6524
# 11 0.70 531.9342
# 12 0.75 549.1961
# 13 0.80 567.3808
# 14 0.85 585.8494
# 15 0.90 604.2120
# 16 0.95 622.2074
# 17 1.00 639.6127

model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=column), sigma=0.55)
model_grnn.pred <- pred_grnn(test_all[, -column], model_grnn)

model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=column), sigma=0.55)
model_grnn.pred <- pred_grnn(test_all[, -column], model_grnn)
model_grnn.predClass <- round(model_grnn.pred)
u_grnn <- union(test_all[,column], model_grnn.predClass)
caret::confusionMatrix(table(true=factor(test_all[,column], u_grnn), predictions=factor(model_grnn.predClass, u_grnn)))


# Accuracy for sig:0.55
# 0.6397059

# Accuracy for sig:0.20 -> 0.4452614

