pkgs <- c('RSNNS', 'caret', 'doParallel', 'foreach', 'grnn', 'pnn')
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
train_target <- decodeClassLabels(unlist(train_all[column]))

predict_class <- function(pred) {
  pred <- data.frame(pred)
  names(pred) <- c("3", "4", "5", "6", "7", "8", "9")
  return(names(pred)[apply(pred, 1, which.max)]) 
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(2, 20, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_target, size=node, maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- predict_class(model_mlp.pred)
  u_mlp <- union(test_all$quality, model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])

# node  accuracy
# Accuracy16   18 0.5637255
# Accuracy17   19 0.5637255
# > cv
# node  accuracy
# Accuracy      2 0.5408497
# Accuracy1     3 0.5187908
# Accuracy2     4 0.5220588
# Accuracy3     5 0.5375817
# Accuracy4     6 0.5122549
# Accuracy5     7 0.5269608
# Accuracy6     8 0.5400327
# Accuracy7     9 0.5457516
# Accuracy8    10 0.5490196
# Accuracy9    11 0.5531046
# Accuracy10   12 0.5449346
# Accuracy11   13 0.5604575
# Accuracy12   14 0.5522876
# Accuracy13   15 0.5498366
# Accuracy14   16 0.5408497
# Accuracy15   17 0.5580065
# Accuracy16   18 0.5637255
# Accuracy17   19 0.5637255
# Accuracy18   20 0.5400327

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv2 <- foreach(node=seq(2, 10, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_target, size=c(18,node), maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- predict_class(model_mlp.pred)
  u_mlp <- union(test_all$quality, model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node2 <- cv2[cv2$accuracy==max(cv2$accuracy), 1:2])

# node  accuracy
# Accuracy1    3 0.5767974
# 
# node  accuracy
# Accuracy     2 0.5604575
# Accuracy1    3 0.5767974
# Accuracy2    4 0.5686275
# Accuracy3    5 0.5669935
# Accuracy4    6 0.5710784
# Accuracy5    7 0.5702614
# Accuracy6    8 0.5661765
# Accuracy7    9 0.5416667
# Accuracy8   10 0.5751634

set.seed(101)
model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_target, size=c(18,node), maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[-column])
model_mlp.predClass <- predict_class(model_mlp.pred)
u_mlp <- union(test_all$quality, model_mlp.predClass)
m <- caret::confusionMatrix(table(true=factor(test_all$quality, u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))