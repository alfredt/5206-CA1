diabetes <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")
#summary(diabetes)

norm <- function(x) {
  return((x -min(x))/(max(x)-min(x)))
}

give_class <- function(x) {
  return(ifelse(x==1, "Yes", "No"))
}

diabetes_norm <- data.frame(lapply(diabetes, norm))
require(caTools)
set.seed(101) 
sample = sample.split(diabetes_norm, SplitRatio = .75)
train = subset(diabetes_norm, sample == TRUE)
test  = subset(diabetes_norm, sample == FALSE)

library(nnet)
model_nnet <- nnet(V9 ~ ., data=train, size=8, maxit=1000)

pred <- predict(model_nnet, test)
  
roundUp <- function(x) {
  return(ifelse(x>0.5, "Yes", "No"))
}

pred_class = roundUp(pred)

actual_class = give_class(test$V9)

table(true=actual_class, predicted=pred_class)

model_nnet2 <- nnet(x=train[,1:8], y=train[,9], data=train, size=8, maxit=1000)

pred2 <- predict(model_nnet, test[,1:8])
pred_class2 = roundUp(pred2)
table(true=actual_class, predicted=pred_class2)


library(RSNNS)

model_mlp10 <- mlp(x=train[,1:8], y=train[,9], size=8, learnFuncParams=c(0.1), maxit=50)
pred3 <- predict(model_mlp10, test[,1:8])
pred_class3 = roundUp(pred3)
table(true=actual_class, predicted=pred_class3)

model_mlp84 <- mlp(x=train[,1:8], y=train[,9], size=c(8,4), learnFuncParams=c(0.1), maxit=50)
pred4 <- predict(model_mlp84, test[,1:8])
pred_class4 = roundUp(pred4)
table(true=actual_class, predicted=pred_class4)

#rbf
model_rbf <- rbf(x=train[,1:8], y=train[,9], size=100, maxit=100, linOut=TRUE)
pred5 <- predict(model_rbf, test[,1:8])
pred_class5 = roundUp(pred5)
table(true=actual_class, predicted=pred_class5)
