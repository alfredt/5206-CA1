pkgs <- c('caret', 'doParallel', 'foreach', 'nnet', 'RSNNS', 'grnn', 'pnn')
lapply(pkgs, require, character.only=T)
registerDoParallel(cores=8)

# Reading of data
data_nnet <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")
data_all <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")

# Data preprocessing
column = ncol(data_nnet)
data_nnet$quality <- as.factor(data_nnet$quality)

# Scaling of data
data_nnet <- data.frame(scale(data_nnet[-column]), data_nnet[column])
data_all <- data.frame(scale(data_all[-column]), data_all[column])

# Split data into 75/25
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
test_class <- as.integer(test_all$V9)

# NNs size/sigma settings
nnet_nodes = 5
mlp_nodes = c(18, 3)
rbf_nodes = 18
grnn_sigma = 0.55
pnn_sigma = 0.2

# NNs predict functions
predict_class <- function(pred) {
  pred <- data.frame(pred)
  names(pred) <- c("3", "4", "5", "6", "7", "8", "9")
  return(names(pred)[apply(pred, 1, which.max)]) 
}

predict_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=grnn::guess(nn, as.matrix(i)))
  }
}

predict_pnn <- function(x, pnn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i=xlst, .combine=rbind) %dopar% {
    c(pred=pnn::guess(pnn, as.matrix(i))$category)
  }
}

# Training & Predictions of NNs models
set.seed(101)
model_nnet <- nnet::nnet(train_nnet$quality ~ ., data=train_nnet, size=nnet_nodes, maxit=1000)
model_nnet.predClass <- as.integer(predict(model_nnet, test_nnet, type="class"))

set.seed(101)
model_mlp <- RSNNS::mlp(x=train_all[-column], y=train_target, size=mlp_nodes, maxit=1000)
model_mlp.pred <- predict(model_mlp, test_all[-column])
model_mlp.predClass <- predict_class(model_mlp.pred)

set.seed(101)
model_rbf <- RSNNS::rbf(x=train_all[-column], y=train_all[column], size=rbf_nodes, maxit=1000)
model_rbf.pred <- predict(model_mlp, test_all[-column])
model_rbf.predClass <- as.integer(round(model_mlp.pred))

set.seed(101)
model_grnn <- grnn::smooth(grnn::learn(train_all, variable.column=column), sigma=grnn_sigma)
model_grnn.pred <- predict_grnn(test_all[, -column], model_grnn)
model_grnn.predClass <- as.integer(round(model_grnn.pred))

set.seed(101)
model_pnn <- pnn::smooth(pnn::learn(train_all, category.column=column), sigma=pnn_sigma)
model_pnn.predClass <- as.integer(predict_pnn(test_all[, -column], model_pnn))

# Predicting ensemble predictions
all_predClass <- data.frame(model_nnet.predClass,
                            model_mlp.predClass,
                            model_rbf.predClass,
                            model_grnn.predClass,
                            model_pnn.predClass)

sum_predClass <- rowSums(all_predClass)
final_predClass <- predict_overall(sum_predClass)
u <- union(test_class, final_predClass)
caret::confusionMatrix(table(true=factor(test_class, u), predictions=factor(final_predClass, u)))
# 
# Confusion Matrix and Statistics
# 
# predictions
# true   1   0
#     1  57  48
#     0  13 138
# 
# Accuracy : 0.7617          
# 95% CI : (0.7047, 0.8126)
# No Information Rate : 0.7266          
# P-Value [Acc > NIR] : 0.1157          
# 
# Kappa : 0.4812          
# Mcnemar's Test P-Value : 1.341e-05       
#                                           
#             Sensitivity : 0.8143          
#             Specificity : 0.7419          
#          Pos Pred Value : 0.5429          
#          Neg Pred Value : 0.9139          
#              Prevalence : 0.2734          
#          Detection Rate : 0.2227          
#    Detection Prevalence : 0.4102          
#       Balanced Accuracy : 0.7781          
#                                           
#        'Positive' Class : 1
