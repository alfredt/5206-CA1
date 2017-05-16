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
