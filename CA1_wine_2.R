pkgs <- c('nnet', 'RSNNS', 'caret', 'doParallel', 'foreach', 'grnn', 'pnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_nnet <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")
data_nnet$quality <- as.factor(data_nnet$quality)

data_all <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")

# PRE-PROCESSING DATA
column = ncol(data_nnet)
data_nnet <- data.frame(scale(data_nnet[-column]), data_nnet[column])
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
train_target <- decodeClassLabels(unlist(train_all[column]))

#######################################
## nnet models
#######################################
set.seed(101) 
model_nnet <- nnet(train_nnet$quality ~ ., data=train_nnet, size=8, maxit=1000)
model_nnet.pred <- predict(model_nnet, test_nnet)
model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
u_nnet <- union(test_nnet$quality, model_nnet.predClass)
caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))

#######################################
## mlp models
#######################################
predict_class <- function(pred) {
  pred <- data.frame(pred)
  names(pred) <- c("3", "4", "5", "6", "7", "8", "9")
  return(names(pred)[apply(pred, 1, which.max)]) 
}

set.seed(101)
model_mlp <- mlp(x=train_all[-column], y=train_target, size=8, maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[-column])
model_mlp.predClass <- predict_class(model_mlp.pred)
u_mlp <- union(test_all$quality, model_mlp.predClass)
caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))

#######################################
## rbf models
#######################################
set.seed(101)
model_rbf <- rbf(x=train_all[-column], y=train_target, size=8, maxit=1000, linOut=TRUE)
model_rbf.pred <- predict(model_rbf, test_all[-column])
model_rbf.predClass = predict_class(model_rbf.pred)
u_rbf <- union(test_all$quality, model_rbf.predClass)
caret::confusionMatrix(table(true=factor(test_all$quality, u_rbf), predictions=factor(model_rbf.predClass, u_rbf)))


