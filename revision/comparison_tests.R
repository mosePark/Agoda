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
SVD_ori <- readRDS("D:/hug-ebd-svd/SVD_ori.rds")

SVD_gen1_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_1.rds")
SVD_gen2_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_1.rds")
SVD_gen1_1_0_1 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_1.rds")

SVD_gen1_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_4.rds")
SVD_gen2_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_4.rds")
SVD_gen1_1_0_4 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_4.rds")

SVD_gen1_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen1_0_7.rds")
SVD_gen2_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen2_0_7.rds")
SVD_gen1_1_0_7 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0_7.rds")

SVD_gen1_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_0.rds")
SVD_gen2_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen2_1_0.rds")
SVD_gen1_1_1_0 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_1_0.rds")

dim = 2


'
baseline
'

# Original Data Reduction
red_or <- SVD_ori$u %*% rbind(diag(SVD_ori$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

# gen1_0_1
red_gen1_0_1 <- SVD_gen1_0_1$u %*% rbind(diag(SVD_gen1_0_1$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_1 <- SVD_gen2_0_1$u %*% rbind(diag(SVD_gen2_0_1$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_1 <- SVD_gen1_1_0_1$u %*% rbind(diag(SVD_gen1_1_0_1$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

# gen1_0_4
red_gen1_0_4 <- SVD_gen1_0_4$u %*% rbind(diag(SVD_gen1_0_4$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_4 <- SVD_gen2_0_4$u %*% rbind(diag(SVD_gen2_0_4$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_4 <- SVD_gen1_1_0_4$u %*% rbind(diag(SVD_gen1_1_0_4$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

# gen1_0_7
red_gen1_0_7 <- SVD_gen1_0_7$u %*% rbind(diag(SVD_gen1_0_7$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_7 <- SVD_gen2_0_7$u %*% rbind(diag(SVD_gen2_0_7$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_7 <- SVD_gen1_1_0_7$u %*% rbind(diag(SVD_gen1_1_0_7$d[1:2]), matrix(0, nrow=1536-2, ncol=2))

# gen1_1_0
red_gen1_1_0 <- SVD_gen1_1_0$u %*% rbind(diag(SVD_gen1_1_0$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_1_0 <- SVD_gen2_1_0$u %*% rbind(diag(SVD_gen2_1_0$d[1:2]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_1_0 <- SVD_gen1_1_1_0$u %*% rbind(diag(SVD_gen1_1_1_0$d[1:2]), matrix(0, nrow=1536-2, ncol=2))



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

# eq test by folds
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

# Ball test by folds
res <- list()

for (i in 1:kfds) {
  cat("\n### Fold", i, "###\n")
  
  current_idx <- folds[[i]]
  
  X_part <- X[current_idx, , drop = FALSE]
  Y_part <- Y[current_idx, , drop = FALSE]
  
  # combined_data <- rbind(X_part, Y_part)
  
  # 검정 수행
  test_result <- Ball::bd.test(X_part, Y_part)
  
  res[[paste0("Fold_", i)]] <- test_result
  
  cat("p-value:", test_result$p.value, "\n")
}

# kernel test by folds
res <- list()

for (i in 1:kfds) {
  cat("kernel test", "\n### Fold", i, "###\n")
  
  current_idx <- folds[[i]]
  
  X_part <- X[current_idx, , drop = FALSE]
  Y_part <- Y[current_idx, , drop = FALSE]
  
  
  # 검정 수행
  sig = med_sigma(X_part, Y_part) # gaussian kernel의 bandwidth
  test_result <- kerTests::kertests(X_part, Y_part, sigma=sig)
  
  res[[paste0("Fold_", i)]] <- test_result
  
  cat("p-value:", test_result$p.value, "\n")
}

# Final

comparison_tests <- function(source = 'quora', X, Y, kfds = 5, method = c("eqdist", "ball", "kernel"), seed = 123) {
  set.seed(seed)
  idx <- which(df$source == source)
  X <- X[idx, , drop = FALSE]
  Y <- Y[idx, , drop = FALSE]
  
  n <- nrow(X)
  folds <- split(sample(1:n), rep(1:kfds, length.out = n))
  
  res <- list()
  
  for (i in 1:kfds) {
    current_idx <- folds[[i]]
    X_part <- X[current_idx, , drop = FALSE]
    Y_part <- Y[current_idx, , drop = FALSE]
    
    if (method == "eqdist") {
      combined_data <- rbind(X_part, Y_part)
      test_result <- energy::eqdist.etest(dist(combined_data), sizes = c(nrow(X_part), nrow(Y_part)), R = 99)
      res[[paste0("Fold_", i)]] <- test_result
      cat("\n### Fold", i, "(eqdist) ###\n")
      print(test_result)
      
    } else if (method == "ball") {
      test_result <- Ball::bd.test(X_part, Y_part)
      res[[paste0("Fold_", i)]] <- test_result
      cat("\n### Fold", i, "(ball) ###\n")
      print(test_result)
      
    } else if (method == "kernel") {
      sig <- med_sigma(X_part, Y_part)
      test_result <- kerTests::kertests(X_part, Y_part, sigma = sig)
      res[[paste0("Fold_", i)]] <- test_result
      cat("\n### Fold", i, "(kernel) ###\n")
      print(test_result)
    }
  }
  return(res)
}



comparison_tests(source='squad_2', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=10, method='eqdist', seed=123)
