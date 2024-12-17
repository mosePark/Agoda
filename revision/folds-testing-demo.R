'
Temp. 0.7 일 때 돌려보기
'

rm(list = ls(all=TRUE))

make_matrix = function(xvec){
  return_mat = matrix(NA, nrow=length(xvec), ncol=1536)
  for (i in 1:length(xvec)){
    x = xvec[i]
    x_clean = gsub("\\[|\\]", "", x)
    x_numeric <- as.numeric(unlist(strsplit(x_clean, ","))) 
    return_mat[i,] = x_numeric
  }
  return(return_mat)
}

set.seed(123)

'
Load Data
'

df <- read.csv("D:/hug-data/hug-ori.csv", header=T)

ori_emb <- readRDS("D:/hug-ebd-svd/ori_emb.rds")

# gen1_emb_0_1 <- readRDS("D:/hug-ebd-svd/gen1_emb_0_1.rds")
# gen2_emb_0_1 <- readRDS("D:/hug-ebd-svd/gen2_emb_0_1.rds")
# gen1_1_emb_0_1 <- readRDS("D:/hug-ebd-svd/gen1_1_emb_0_1.rds")
# 
# SVD_ori <- readRDS("D:/hug-ebd-svd/SVD_ori.rds")
# SVD_gen1_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_1.rds")
# SVD_gen2_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_1.rds")
# SVD_gen1_1_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_1.rds")
# 
# SVD_gen1_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_4.rds")
# SVD_gen2_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_4.rds")
# SVD_gen1_1_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_4.rds")

SVD_gen1_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_7.rds")
SVD_gen2_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_7.rds")
SVD_gen1_1_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_7.rds")

# SVD_gen1_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0.rds")
# SVD_gen2_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen2_1_0.rds")
# SVD_gen1_1_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_1_0.rds")

dim = 2


'
baseline
'

red_or = SVD_ori$u%*%rbind(diag(SVD_ori$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

red_gen1_0_7 = SVD_gen1_0_7$u%*%rbind(diag(SVD_gen1_0_7$d[1:2]), matrix(0, nrow=1536-2, ncol=2)) # red_gen1_0_1 = gen1_emb_0_7%*%SVD_gen1_0_7$v[,1:2]
red_gen2_0_7 = SVD_gen2_0_7$u%*%rbind(diag(SVD_gen2_0_7$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

library(energy)
energy::eqdist.etest(dist(rbind(red_gen1_0_7[1:1000,], red_gen2_0_7[1:1000,])), sizes=c(1000, 1000), R=99)

library(Ball)
Ball::bd.test(red_gen1_0_7[1:1000,], red_gen2_0_7[1:1000,])

library(kerTests)
sig = med_sigma(red_gen1_0_7[1:1000,], red_gen2_0_7[1:1000,]) # gaussian kernel의 bandwidth
kerTests::kertests(red_gen1_0_7[1:1000,], red_gen2_0_7[1:1000,], sigma=sig)


'
5-folds, quora
'

kfds = 5

idx <- which(df$source == 'quora')

X = red_gen1_0_7[idx,]
Y = red_gen2_0_7[idx,]

set.seed(123)
n <- nrow(X)

folds <- split(sample(1:n), rep(1:kfds, length.out = n))

# testing by folds
res <- list()

for (i in 1:kfds) {
  cat("\n### Fold", i, "###\n")
  
  current_idx <- folds[[i]]
  
  X_part <- X[current_idx, , drop = FALSE]
  Y_part <- Y[current_idx, , drop = FALSE]
  
  combined_data <- rbind(X_part, Y_part)
  
  # 검정 수행
  test_result <- energy::eqdist.etest(dist(combined_data), 
                                      sizes = c(nrow(X_part), nrow(Y_part)), 
                                      R = 99)
  
  res[[paste0("Fold_", i)]] <- test_result
  
  cat("p-value:", test_result$p.value, "\n")
}

