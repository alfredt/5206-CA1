pkgs <- c('caret', 'doParallel', 'foreach', 'grnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")

give_class <- function(x) {
  return(ifelse(x==1, "Yes", "No"))
}

roundUp <- function(x) {
  return(ifelse(x>0.5, "Yes", "No"))
}

# PRE-PROCESSING DATA
length = ncol(data)
X <- data[-length]
st.X <- scale(X)
Y <- data[length]
data_scaled <- data.frame(st.X, Y)

require(caTools)
set.seed(101)
sample = sample.split(data_scaled, SplitRatio=.8)
train = subset(data_scaled, sample==TRUE)
test  = subset(data_scaled, sample==FALSE)
testClass <- give_class(test$V9)

# DEFINE A FUNCTION TO SCORE GRNN
pred_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    return(pred=guess(nn, as.matrix(i)))
  }
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(sig=seq(0.2, 1, 0.05), .combine=rbind) %dopar% {
  set.seed(101)
  model_grnn <- smooth(learn(train, variable.column=ncol(train)), sigma=sig)
  model_grnn.pred <- pred_grnn(test[, -ncol(test)], model_grnn)
  test.sse <- sum((test[, ncol(test)] - model_grnn.pred)^2)
  data.frame(sig, sse=test.sse)
}

cat("\n### BEST SIGMA WITH THE LOWEST SSE ###\n")
print(best.sig <- cv[cv$sse==min(cv$sse), 1])

# SCORE THE WHOLE DATASET WITH GRNN
set.seed(101)
model_grnn <- smooth(learn(train, variable.column=ncol(train)), sigma=0.85)
model_grnn.pred <- pred_grnn(test[, -ncol(test)], model_grnn)
model_grnn.predClass <- roundUp(model_grnn.pred)
caret::confusionMatrix(table(true=testClass, predicted=model_grnn.predClass))


