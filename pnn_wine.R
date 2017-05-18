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

data.frame(sigma, accuracy=a)

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(sigma=seq(0.2, 1, 0.05), .combine=rbind) %dopar% {
  set.seed(101)
  model_pnn <- pnn::smooth(pnn::learn(train_all, category.column=column), sigma=sigma)
  model_pnn.pred <- predict_pnn(test_all[, -column], model_pnn)
  u_pnn <- union(test_all[column], model_pnn.pred)
  m <- caret::confusionMatrix(table(true=factor(test_all[[column]], u_pnn), predictions=factor(model_pnn.pred, u_pnn)))
  a <- m$overall[1]
  data.frame(sigma, accuracy=a)
}

cat("\n### BEST SIGMA WITH THE HIGHEST ACCURACY ###\n")
print(best.sigma <- cv[cv$accuracy==max(cv$accuracy), 1:2])

# sigma  accuracy
# Accuracy   0.2 0.6094771
# 
# sigma  accuracy
# Accuracy    0.20 0.6094771
# Accuracy1   0.25 0.6045752
# Accuracy2   0.30 0.5964052
# Accuracy3   0.35 0.5874183
# Accuracy4   0.40 0.5735294
# Accuracy5   0.45 0.5661765
# Accuracy6   0.50 0.5473856
# Accuracy7   0.55 0.5285948
# Accuracy8   0.60 0.5147059
# Accuracy9   0.65 0.4918301
# Accuracy10  0.70 0.4852941
# Accuracy11  0.75 0.4648693
# Accuracy12  0.80 0.4501634
# Accuracy13  0.85 0.4248366
# Accuracy14  0.90 0.4191176
# Accuracy15  0.95 0.4133987
# Accuracy16  1.00 0.4027778
