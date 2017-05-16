pkgs <- c('nnet', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_nnet <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")
data_nnet$quality <- as.factor(data_nnet$quality)

# PRE-PROCESSING DATA
column = ncol(data_nnet)
data_nnet <- data.frame(scale(data_nnet[-column]), data_nnet[column])

require("caTools")
set.seed(101) 
sample = sample.split(data_nnet, SplitRatio = .75)
train_nnet = subset(data_nnet, sample == TRUE)
test_nnet  = subset(data_nnet, sample == FALSE)

#######################################
## nnet models
#######################################
# set.seed(101) 
# model_nnet <- nnet(train_nnet$quality ~ ., data=train_nnet, size=8, maxit=1000)
# model_nnet.pred <- predict(model_nnet, test_nnet)
# model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
# u_nnet <- union(test_nnet$quality, model_nnet.predClass)
# m <- caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(5, 15, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_nnet <- nnet::nnet(train_nnet$quality ~ ., data=train_nnet, size=node, maxit=1000)
  model_nnet.predClass <- predict(model_nnet, test_nnet, type="class")
  u_nnet <- union(test_nnet$quality, model_nnet.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_nnet$quality, u_nnet), predictions=factor(model_nnet.predClass, u_nnet)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])

# node  accuracy
# Accuracy     5 0.5596405
# Accuracy4    9 0.5596405

# node  accuracy
# Accuracy      5 0.5596405
# Accuracy1     6 0.5392157
# Accuracy2     7 0.5367647
# Accuracy3     8 0.5588235
# Accuracy4     9 0.5596405
# Accuracy5    10 0.5375817
# Accuracy6    11 0.5310458
# Accuracy7    12 0.5294118
# Accuracy8    13 0.5441176
# Accuracy9    14 0.5473856
# Accuracy10   15 0.5531046

