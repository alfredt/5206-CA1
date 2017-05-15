pkgs <- c('nnet', 'RSNNS', 'caret', 'doParallel', 'foreach', 'grnn', 'pnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_nnet <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")
data_nnet$V9 <- as.factor(data_nnet$V9)

data_all <- read.csv(file="Diabetes.csv", head=FALSE, sep=",")

# PRE-PROCESSING DATA
data_nnet <- data.frame(scale(data_nnet[-ncol(data_nnet)]), data_nnet[ncol(data_nnet)])
data_all <- data.frame(scale(data_all[-ncol(data_all)]), data_all[ncol(data_all)])

require("caTools")
set.seed(101) 
sample = sample.split(data_nnet, SplitRatio = .75)
train_nnet = subset(data_nnet, sample == TRUE)
test_nnet  = subset(data_nnet, sample == FALSE)

set.seed(101) 
sample = sample.split(data_all, SplitRatio = .75)
train_all = subset(data_all, sample == TRUE)
test_all  = subset(data_all, sample == FALSE)

#######################################
## nnet models
#######################################
set.seed(101) 
model_nnet <- nnet(V9 ~ ., data=train_nnet, size=8, maxit=1000)
model_nnet.pred <- predict(model_nnet, test_nnet)
model_nnet.predClass <-  predict(model_nnet, test_nnet, type="class")
caret::confusionMatrix(table(true=test_nnet$V9, predictions=model_nnet.predClass))

#######################################
## mlp models
#######################################
set.seed(101)
model_mlp <- mlp(x=train_all[,1:8], y=train_all[,9], size=8, maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[,1:8])
model_mlp.predClass <- round(model_mlp.pred)
caret::confusionMatrix(table(true=test_all$V9, predicted=model_mlp.predClass))

#######################################
## rbf models
#######################################
set.seed(101)
model_rbf <- rbf(x=train_all[,1:8], y=train_all[,9], size=8, maxit=1000, linOut=TRUE)
model_rbf.pred <- predict(model_rbf, test_all[,1:8])
model_rbf.predClass = round(model_rbf.pred)
caret::confusionMatrix(table(true=test_all$V9, predicted=model_rbf.predClass))

#######################################
## grnn models
#######################################
# DEFINE FUNCTIONS TO SCORE GRNN
predict_grnn <- function(x, grnn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=grnn::guess(grnn, as.matrix(i)))
  }
}

set.seed(101)
model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=ncol(train_all)), sigma=0.85)
model_grnn.pred <- predict_grnn(test_all[, -ncol(test_all)], model_grnn)
model_grnn.predClass <- round(model_grnn.pred)
u <- union(test_all$V9, model_grnn.predClass)
caret::confusionMatrix(table(true=factor(test_all$V9, u), predicted=factor(model_grnn.predClass, u)))

#######################################
## pnn models
#######################################
# DEFINE FUNCTIONS TO SCORE PNN
predict_pnn <- function(x, pnn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=pnn::guess(pnn, as.matrix(i))$category)
  }
}

set.seed(101)
model_pnn <- pnn::smooth(pnn::learn(train_all, category.column=ncol(train_all)), sigma=0.85)
model_pnn.predClass <- predict_pnn(test_all[, -ncol(test_all)], model_pnn)
u_p <- union(test_all$V9, model_pnn.pred)
caret::confusionMatrix(table(true=factor(test_all$V9, u_p), predicted=factor(model_pnn.predClass, u_p)))


