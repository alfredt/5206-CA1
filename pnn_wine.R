pkgs <- c('caret', 'doParallel', 'foreach', 'pnn')
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
predict_pnn <- function(x, pnn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=pnn::guess(pnn, as.matrix(i))$category)
  }
}

#TRAINING & TESTING
sigma <- 0.55

set.seed(101)
model_grnn <- grnn::smooth(grnn::learn(train, variable.column=ncol(train)), sigma=sigma)
model_grnn.pred <- predict_grnn(test[, -ncol(test)], model_grnn)
model_grnn.predClass <- round(model_grnn.pred)
u <- union(testClass, model_grnn.predClass)
caret::confusionMatrix(table(true=factor(testClass, u), predicted=factor(model_grnn.predClass, u)))

set.seed(101)
model_pnn <- pnn::smooth(pnn::learn(train, category.column=ncol(train)), sigma=sigma)
model_pnn.pred <- predict_pnn(test[, -ncol(test)], model_pnn)
model_pnn.predClass <- model_pnn.pred[,-1]
u <- union(testClass, model_pnn.pred)
caret::confusionMatrix(table(true=factor(testClass, u), predicted=factor(model_pnn.pred, u)))
