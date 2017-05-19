pkgs <- c('caret', 'doParallel', 'foreach', 'grnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")

# PRE-PROCESSING DATA
length = ncol(data)
X <- data[-length]
st.X <- scale(X)
Y <- data[length]
data_scaled <- data.frame(st.X, Y)

require(caTools)
set.seed(101)
sample = sample.split(data_scaled, SplitRatio=.75)
train = subset(data_scaled, sample==TRUE)
test  = subset(data_scaled, sample==FALSE)
testClass <- test[,ncol(test)]

# DEFINE A FUNCTION TO SCORE GRNN
pred_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=grnn::guess(nn, as.matrix(i)))
  }
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(sig=seq(0.2, 1, 0.05), .combine=rbind) %dopar% {
  set.seed(101)
  model_grnn <- grnn::smooth(grnn::learn(train, variable.column=ncol(train)), sigma=sig)
  model_grnn.pred <- pred_grnn(test[, -ncol(test)], model_grnn)
  test.sse <- sum((test[, ncol(test)] - model_grnn.pred)^2)
  data.frame(sig, sse=test.sse)
}

cat("\n### BEST SIGMA WITH THE LOWEST SSE ###\n")
print(best.sig <- cv[cv$sse==min(cv$sse), 1])

# SCORE THE WHOLE DATASET WITH GRNN
set.seed(101)
model_grnn <- grnn::smooth(grnn::learn(train, variable.column=ncol(train)), sigma=0.55)
model_grnn.pred <- pred_grnn(test[, -ncol(test)], model_grnn)
model_grnn.predClass <- round(model_grnn.pred)
u <- union(testClass, model_grnn.predClass)
caret::confusionMatrix(table(true=factor(testClass, u), predicted=factor(model_grnn.predClass, u)))

set.seed(101)
model_grnn2 <- smooth(learn(train, variable.column=ncol(train)), sigma=0.05)
model_grnn2.pred <- pred_grnn(test[, -ncol(test)], model_grnn2)
model_grnn2.predClass <- round(model_grnn2.pred)
u2 <- union(testClass, model_grnn2.predClass)
caret::confusionMatrix(table(true=factor(testClass, u2), predicted=factor(model_grnn2.predClass, u2)))
