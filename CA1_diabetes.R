diabetes <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")
#summary(diabetes)
library(caret)

norm <- function(x) {
  return((x -min(x))/(max(x)-min(x)))
}

give_class <- function(x) {
  return(ifelse(x==1, "Yes", "No"))
}

roundUp <- function(x) {
  return(ifelse(x>0.5, "Yes", "No"))
}

diabetes_norm <- data.frame(lapply(diabetes, norm))
require(caTools)
set.seed(101) 
sample = sample.split(diabetes_norm, SplitRatio = .75)
train = subset(diabetes_norm, sample == TRUE)
test  = subset(diabetes_norm, sample == FALSE)
testClass <- give_class(test$V9)

library(nnet)
#######################################
## nnet models
#######################################
set.seed(101) 
model_nnet_1 <- nnet(V9 ~ ., data=train, size=8, maxit=1000)
model_nnet_1.pred <- predict(model_nnet_1, test)
model_nnet_1.predClass <- roundUp(model_nnet_1.pred)
caret::confusionMatrix(table(true=testClass, predictions=model_nnet_1.predClass))

set.seed(101)
model_nnet_2 <- nnet(x=train[,1:8], y=train[,9], data=train, size=8, maxit=1000)
model_nnet_2.pred <- predict(model_nnet_2, test[,1:8])
model_nnet_2.predClass <- roundUp(model_nnet_2.pred)
caret::confusionMatrix(table(true=testClass, predicted=model_nnet_2.predClass))

library(RSNNS)
#######################################
## mlp models
#######################################
set.seed(101)
model_mlp_1 <- mlp(x=train[,1:8], y=train[,9], size=8, maxit=1000)
model_mlp_1.pred <- predict(model_mlp_1, test[,1:8])
model_mlp_1.predClass <- roundUp(model_mlp_1.pred)
caret::confusionMatrix(table(true=testClass, predicted=model_mlp_1.predClass))

set.seed(101)
model_mlp_2 <- mlp(x=train[,1:8], y=train[,9], size=c(3,3), learnFuncParams=c(0.1), maxit=1000)
model_mlp_2.pred <- predict(model_mlp_2, test[,1:8])
model_mlp_2.predClass <- roundUp(model_mlp_2.pred)
caret::confusionMatrix(table(true=testClass, predicted=model_mlp_2.predClass))

#######################################
## rbf models
#######################################
set.seed(101)
model_rbf <- rbf(x=train[,1:8], y=train[,9], size=8, maxit=1000, linOut=TRUE)
model_rbf.pred <- predict(model_rbf, test[,1:8])
model_rbf.predClass = roundUp(model_rbf.pred)
caret::confusionMatrix(table(true=testClass, predicted=model_rbf.predClass))

library(grnn)
#######################################
## grnn models
#######################################
length = ncol(train)
model_grnn <- grnn::learn(train, variable.column=length)
model_grnn <- grnn::smooth(model_grnn, sigma=0.5)
test_grnn <- test[,1:length-1]
model_grnn.pred <- matrix(-1, nrow=nrow(test), ncol=1)

for(i in 1:nrow(test)) {
  vec <- as.matrix(test_grnn[i,])
  pred <- grnn::guess(model_grnn, vec)
  
  if(is.nan(pred)) {
    cat("Entry: ", i, " Generated NaN result!\n")
  }
  else {
    model_grnn.pred[i] <- pred
  }
}

model_grnn.predClass = round(model_grnn.pred)
model_grnn.predClass = roundUp(model_grnn.predClass)
caret::confusionMatrix(table(true=testClass, predicted=model_grnn.predClass))


length = ncol(diabetes_norm)
model_grnn2 <- grnn::learn(diabetes_norm, variable.column=length)
model_grnn2 <- grnn::smooth(model_grnn2, sigma=0.5)
test_grnn2 <- diabetes_norm[,1:length-1]
model_grnn2.pred <- matrix(-1, nrow=nrow(diabetes_norm), ncol=1)

for(i in 1:nrow(model_grnn2.pred)) {
  vec <- as.matrix(test_grnn2[i,])
  pred <- grnn::guess(model_grnn, vec)
  
  if(is.nan(pred)) {
    cat("Entry: ", i, " Generated NaN result!\n")
  }
  else {
    model_grnn2.pred[i] <- pred
  }
}

model_grnn2.predClass = round(model_grnn2.pred)
model_grnn2.predClass = roundUp(model_grnn2.predClass)
caret::confusionMatrix(table(true=testClass, predicted=model_grnn2.predClass))
