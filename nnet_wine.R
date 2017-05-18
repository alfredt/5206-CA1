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
cv <- foreach(node=seq(2, 20, 1), .combine=rbind) %dopar% {
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

### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###
# node  accuracy
# Accuracy3    5 0.5596405
# Accuracy6    8 0.5596405
# > cv
# node  accuracy
# Accuracy      2 0.5187908
# Accuracy1     3 0.5424837
# Accuracy2     4 0.5359477
# Accuracy3     5 0.5596405
# Accuracy4     6 0.5424837
# Accuracy5     7 0.5367647
# Accuracy6     8 0.5596405
# Accuracy7     9 0.5555556
# Accuracy8    10 0.5351307
# Accuracy9    11 0.5294118
# Accuracy10   12 0.5310458
# Accuracy11   13 0.5449346
# Accuracy12   14 0.5457516
# Accuracy13   15 0.5514706
# Accuracy14   16 0.5359477
# Accuracy15   17 0.5571895
# Accuracy16   18 0.5228758
# Accuracy17   19 0.5089869
# Accuracy18   20 0.5359477

