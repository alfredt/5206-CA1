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
cv <- foreach(node=seq(2, 15, 1), .combine=rbind) %dopar% {
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
# Accuracy10   12 0.5555556
# 
# node  accuracy
# Accuracy      2 0.5334967
# Accuracy1     3 0.5416667
# Accuracy2     4 0.5253268
# Accuracy3     5 0.5400327
# Accuracy4     6 0.5187908
# Accuracy5     7 0.5334967
# Accuracy6     8 0.5416667
# Accuracy7     9 0.5424837
# Accuracy8    10 0.5531046
# Accuracy9    11 0.5334967
# Accuracy10   12 0.5555556
# Accuracy11   13 0.5433007
# Accuracy12   14 0.5498366
# Accuracy13   15 0.5465686

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv2 <- foreach(node=seq(2, 8, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_target, size=c(12,node), maxit=1000)
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
# Accuracy4    6 0.5825163
# 
# node  accuracy
# Accuracy     2 0.5588235
# Accuracy1    3 0.5433007
# Accuracy2    4 0.5686275
# Accuracy3    5 0.5596405
# Accuracy4    6 0.5825163
# Accuracy5    7 0.5522876
# Accuracy6    8 0.5555556