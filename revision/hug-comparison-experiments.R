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

set.seed(123)


'
Load Data
'

df <- read.csv("D:/hug-data/hug-ori.csv", header=T)

# ori_emb <- readRDS("D:/hug-ebd-svd/ori_emb.rds")

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

SVD_gen1_1_5 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_5.rds")
SVD_gen2_1_5 <- readRDS("D:/hug-ebd-svd/SVD_gen2_1_5.rds")
SVD_gen1_1_1_5 <- readRDS("D:/hug-ebd-svd/SVD_gen1_1_1_5.rds")

'
reduction
'

dim = 2

# Original Data Reduction
red_or <- SVD_ori$u %*% rbind(diag(SVD_ori$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# 0.1
red_gen1_0_1 <- SVD_gen1_0_1$u %*% rbind(diag(SVD_gen1_0_1$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_1 <- SVD_gen2_0_1$u %*% rbind(diag(SVD_gen2_0_1$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_1 <- SVD_gen1_1_0_1$u %*% rbind(diag(SVD_gen1_1_0_1$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# 0.4
red_gen1_0_4 <- SVD_gen1_0_4$u %*% rbind(diag(SVD_gen1_0_4$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_4 <- SVD_gen2_0_4$u %*% rbind(diag(SVD_gen2_0_4$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_4 <- SVD_gen1_1_0_4$u %*% rbind(diag(SVD_gen1_1_0_4$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# 0.7
red_gen1_0_7 <- SVD_gen1_0_7$u %*% rbind(diag(SVD_gen1_0_7$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_0_7 <- SVD_gen2_0_7$u %*% rbind(diag(SVD_gen2_0_7$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_0_7 <- SVD_gen1_1_0_7$u %*% rbind(diag(SVD_gen1_1_0_7$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# 1.0
red_gen1_1_0 <- SVD_gen1_1_0$u %*% rbind(diag(SVD_gen1_1_0$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_1_0 <- SVD_gen2_1_0$u %*% rbind(diag(SVD_gen2_1_0$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_1_0 <- SVD_gen1_1_1_0$u %*% rbind(diag(SVD_gen1_1_1_0$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# 1.5
red_gen1_1_5 <- SVD_gen1_1_5$u %*% rbind(diag(SVD_gen1_1_5$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen2_1_5 <- SVD_gen2_1_5$u %*% rbind(diag(SVD_gen2_1_5$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))
red_gen1_1_1_5 <- SVD_gen1_1_1_5$u %*% rbind(diag(SVD_gen1_1_1_5$d[1:dim]), matrix(0, nrow=1536-2, ncol=2))

# cnn_news, quora, squad_2 

'
temp 0.1
'

## eqdist ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='eqdist', seed=7)

## ball ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_1, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_gen1_0_1, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_1, kfds=5, method='ball', seed=7)

'
temp 0.4
'

## eqdist ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='quora', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='squad_2', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='cnn_news', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='eqdist', seed=7)

## ball ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='quora', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='squad_2', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_4, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='cnn_news', X=red_gen1_0_4, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)

comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_4, kfds=5, method='ball', seed=7)




'
temp 0.7
'

## eqdist ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='eqdist', seed=7)

## ball ################################
# quora
comparison_tests(source='quora', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='quora', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)

# squad_2
comparison_tests(source='squad_2', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='squad_2', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)

# cnn_news
comparison_tests(source='cnn_news', X=red_gen1_0_7, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen2_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_gen1_0_7, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)
comparison_tests(source='cnn_news', X=red_or, Y=red_gen1_1_0_7, kfds=5, method='ball', seed=7)





