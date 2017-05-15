data <- read.csv(file="winequality-white.csv", head=TRUE, sep=",")

data$quality <- as.factor(data$quality)

# PRE-PROCESSING DATA
data_scaled <- data.frame(scale(data[-ncol(data)]), data[ncol(data)])

require(caTools)
set.seed(101)
sample = sample.split(data_scaled, SplitRatio = .75)
train = subset(data_scaled, sample == TRUE)
test  = subset(data_scaled, sample == FALSE)

len_col = ncol(data)

train_x <- train[-len_col]
train_y <- train[len_col]

library(nnet)
#######################################
## nnet models
#######################################
set.seed(101)
model_nnet_2 <- nnet(x=train_x, y=train_y, data=train, size=5, maxit=1000, softmax=TRUE)
model_nnet_2.pred <- predict(model_nnet_2, test)

library(RSNNS)
#######################################
## mlp models
#######################################
set.seed(101)
model_mlp_1 <- mlp(x=train_x, y=train_y, size=8, maxit=1000)
model_mlp_1.pred <- predict(model_mlp_1, test_x)
