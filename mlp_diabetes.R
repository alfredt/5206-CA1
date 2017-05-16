pkgs <- c('RSNNS', 'caret', 'doParallel', 'foreach')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

data_all <- read.csv(file="Diabetes.csv", head=TRUE, sep=",")

# PRE-PROCESSING DATA
column = ncol(data_all)
data_all <- data.frame(scale(data_all[-ncol(data_all)]), data_all[ncol(data_all)])

require("caTools")
set.seed(101) 
sample = sample.split(data_all, SplitRatio = .75)
train_all = subset(data_all, sample == TRUE)
test_all  = subset(data_all, sample == FALSE)
train_target <- decodeClassLabels(unlist(train_all[column]))

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(node=seq(2, 15, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_all[column], size=node, maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- round(model_mlp.pred)
  u_mlp <- union(test_all[column], model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node <- cv[cv$accuracy==max(cv$accuracy), 1:2])
# node  accuracy
# Accuracy1    3 0.7568627

# Accuracy      2 0.7529412
# Accuracy1     3 0.7568627
# Accuracy2     4 0.7294118
# Accuracy3     5 0.6823529
# Accuracy4     6 0.7176471
# Accuracy5     7 0.7137255
# Accuracy6     8 0.7019608
# Accuracy7     9 0.6862745
# Accuracy8    10 0.7176471
# Accuracy9    11 0.7372549
# Accuracy10   12 0.7137255
# Accuracy11   13 0.6941176
# Accuracy12   14 0.6823529
# Accuracy13   15 0.6784314

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv2 <- foreach(node=seq(2, 8, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_all[column], size=c(3, node), maxit=1000)
  model_mlp.pred <- predict(model_mlp, test_all[-column])
  model_mlp.predClass <- round(model_mlp.pred)
  u_mlp <- union(test_all[column], model_mlp.predClass)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_mlp), predictions=factor(model_mlp.predClass, u_mlp)))
  a <- m$overall[1]
  data.frame(node, accuracy=a)
}

cat("\n### BEST # OF HIDDEN NODES WITH THE HIGHEST ACCURACY ###\n")
print(best.node2 <- cv2[cv2$accuracy==max(cv2$accuracy), 1:2])
# node  accuracy
# Accuracy2    4 0.7607843

# node  accuracy
# Accuracy     2 0.7411765
# Accuracy1    3 0.7137255
# Accuracy2    4 0.7607843
# Accuracy3    5 0.7568627
# Accuracy4    6 0.7372549
# Accuracy5    7 0.7450980
# Accuracy6    8 0.7333333
