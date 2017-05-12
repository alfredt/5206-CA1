pkgs <- c('MASS', 'doParallel', 'foreach', 'grnn')
lapply(pkgs, require, character.only = T)
registerDoParallel(cores = 8)

data(Boston)
# PRE-PROCESSING DATA 
X <- Boston[-14]
st.X <- scale(X)
Y <- Boston[14]
boston <- data.frame(st.X, Y)

# SPLIT DATA SAMPLES
set.seed(2013)
rows <- sample(1:nrow(boston), nrow(boston) - 200)
set1 <- boston[rows, ]
set2 <- boston[-rows, ]

# DEFINE A FUNCTION TO SCORE GRNN
pred_grnn <- function(x, nn){
  xlst <- split(x, 1:nrow(x))
  pred <- foreach(i = xlst, .combine = rbind) %dopar% {
    data.frame(pred = guess(nn, as.matrix(i)), i, row.names = NULL)
  }
}

# SEARCH FOR THE OPTIMAL VALUE OF SIGMA BY THE VALIDATION SAMPLE
cv <- foreach(s = seq(0.2, 1, 0.05), .combine = rbind) %dopar% {
  grnn <- smooth(learn(set1, variable.column = ncol(set1)), sigma = s)
  pred <- pred_grnn(set2[, -ncol(set2)], grnn)
  test.sse <- sum((set2[, ncol(set2)] - pred$pred)^2)
  data.frame(s, sse = test.sse)
}

cat("\n### SSE FROM VALIDATIONS ###\n")
print(cv)
jpeg('grnn_cv.jpeg', width = 800, height = 400, quality = 100)
with(cv, plot(s, sse, type = 'b'))

cat("\n### BEST SIGMA WITH THE LOWEST SSE ###\n")
print(best.s <- cv[cv$sse == min(cv$sse), 1])

# SCORE THE WHOLE DATASET WITH GRNN
final_grnn <- smooth(learn(set1, variable.column = ncol(set1)), sigma = best.s)
pred_all <- pred_grnn(boston[, -ncol(set2)], final_grnn)
jpeg('grnn_fit.jpeg', width = 800, height = 400, quality = 100)
plot(pred_all$pred, boston$medv) 
dev.off()