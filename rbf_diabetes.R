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
cv <- foreach(node=seq(9, 20, 1), .combine=rbind) %dopar% {
  set.seed(101)
  model_mlp <- RSNNS::rbf(x=train_all[-column], y=train_all[column], size=node, maxit=1000)
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
# Accuracy6   15 0.7882353
# 
# node  accuracy
# Accuracy      9 0.7725490
# Accuracy1    10 0.7803922
# Accuracy2    11 0.7803922
# Accuracy3    12 0.7803922
# Accuracy4    13 0.7725490
# Accuracy5    14 0.7803922
# Accuracy6    15 0.7882353
# Accuracy7    16 0.7843137
# Accuracy8    17 0.7764706
# Accuracy9    18 0.6274510
# Accuracy10   19 0.7725490
# Accuracy11   20 0.7647059

