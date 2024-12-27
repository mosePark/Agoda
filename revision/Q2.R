###############################################################################
'
function
'
###############################################################################

hug_tests <- function(X, Y, source = "quora", X_name = "X", Y_name = "Y", kfds = 5, method = c("eqdist", "ball", "kernel"), seed = 123) {
  set.seed(seed)
  method <- match.arg(method)
  
  n <- nrow(X)
  if (nrow(X) != nrow(Y)) {
    stop("X와 Y의 행(샘플) 수가 일치해야 합니다.")
  }
  
  folds <- split(sample(1:n), rep(1:kfds, length.out = n))
  res <- list()
  
  generate_filename <- function(X_name, Y_name, method, seed) {
    paste0(
      "comparison_tests_",
      X_name, "_",
      Y_name, "_",
      method, "_",
      seed
    )
  }
  
  output_file <- generate_filename(X_name, Y_name, method, seed)
  
  # 테이블을 저장할 데이터 프레임 초기화
  results_table <- matrix("", nrow = kfds, ncol = 10, dimnames = list(
    paste0("fold", 1:kfds),
    c("0.1 vs 0.4", "0.1 vs 0.7", "0.1 vs 1.0", "0.1 vs 1.5",
      "0.4 vs 0.7", "0.4 vs 1.0", "0.4 vs 1.5",
      "0.7 vs 1.0", "0.7 vs 1.5", "1.0 vs 1.5")
  ))
  
  output_content <- capture.output({
    cat("==== Comparison Tests ====\n")
    cat("Method   :", method, "\n")
    cat("X_name   :", X_name, "\n")
    cat("Y_name   :", Y_name, "\n")
    cat("k-fold   :", kfds, "\n")
    cat("Seed     :", seed, "\n")
    cat("Total N  :", n, "\n")
    cat("Source   :", source, "\n\n")
    
    cat("==== Fold Indices ====\n")
    for (i in seq_len(kfds)) {
      cat("Fold", i, ":", paste(folds[[i]], collapse = ", "), "\n")
    }
    cat("\n==== Test Results ====\n")
    
    for (i in seq_len(kfds)) {
      current_idx <- folds[[i]]
      X_part <- X[current_idx, , drop = FALSE]
      Y_part <- Y[current_idx, , drop = FALSE]
      
      if (method == "eqdist") {
        combined_data <- rbind(X_part, Y_part)
        test_result <- energy::eqdist.etest(
          dist(combined_data), 
          sizes = c(nrow(X_part), nrow(Y_part)), 
          R = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, ] <- paste0("p=", round(test_result$p.value, 3))
        cat("\n### Fold", i, "(eqdist) ###\n")
        print(test_result)
      } else if (method == "ball") {
        test_result <- Ball::bd.test(
          X_part, 
          Y_part, 
          num.permutations = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, ] <- paste0("p=", round(test_result$p.value, 3))
        cat("\n### Fold", i, "(ball) ###\n")
        print(test_result)
      } else if (method == "kernel") {
        sig <- med_sigma(X_part, Y_part)
        test_result <- kerTests::kertests(
          X_part, 
          Y_part, 
          sigma = sig
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, ] <- paste0("p=", round(test_result$p.value, 3))
        cat("\n### Fold", i, "(kernel) ###\n")
        print(test_result)
      }
    }
  })
  
  writeLines(output_content, con = paste0(output_file, ".txt"))
  cat("Saved output to file:", paste0(output_file, ".txt"), "\n")
  
  # 테이블 CSV로 저장
  write.csv(as.data.frame(results_table), file = paste0(output_file, ".csv"), row.names = TRUE)
  cat("Saved table to CSV file:", paste0(output_file, ".csv"), "\n")
  
  return(res)
}

ago_tests <- function(X, Y, source = "agoda", X_name = "X", Y_name = "Y", kfds = 10, method = c("eqdist", "ball", "kernel"), seed = 7) {
  
  set.seed(seed)
  method <- match.arg(method)
  
  n <- nrow(X)
  if(nrow(X) != nrow(Y)) {
    stop("X와 Y의 행(샘플) 수가 일치해야 합니다.")
  }
  
  folds <- split(sample(1:n), rep(1:kfds, length.out = n))
  
  res <- list()
  
  generate_filename <- function(X_name, Y_name, method, seed) {
    paste0(
      "comparison_tests_",
      X_name, "_",
      Y_name, "_",
      method, "_",
      seed
    )
  }
  
  output_file <- generate_filename(X_name, Y_name, method, seed)
  
  # 테이블을 저장할 데이터 프레임 초기화
  results_table <- matrix("", nrow = kfds, ncol = 1, dimnames = list(
    paste0("fold", 1:kfds),
    c("p-value")
  ))
  
  output_content <- capture.output({
    cat("==== Comparison Tests ====\n")
    cat("Method   :", method, "\n")
    cat("X_name   :", X_name, "\n")
    cat("Y_name   :", Y_name, "\n")
    cat("k-fold   :", kfds, "\n")
    cat("Seed     :", seed, "\n")
    cat("Total N  :", n, "\n")
    cat("Source   :", source, "\n\n")
    
    cat("==== Fold Indices ====\n")
    for (i in seq_len(kfds)) {
      cat("Fold", i, ":", paste(folds[[i]], collapse = ", "), "\n")
    }
    cat("\n==== Test Results ====\n")
    
    for (i in seq_len(kfds)) {
      current_idx <- folds[[i]]
      X_part <- X[current_idx, , drop = FALSE]
      Y_part <- Y[current_idx, , drop = FALSE]
      
      if (method == "eqdist") {
        
        combined_data <- rbind(X_part, Y_part)
        test_result <- energy::eqdist.etest(
          dist(combined_data), 
          sizes = c(nrow(X_part), nrow(Y_part)), 
          R = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, "p-value"] <- round(test_result$p.value, 3)
        cat("\n### Fold", i, "(eqdist) ###\n")
        print(test_result)
        
      } else if (method == "ball") {
        
        test_result <- Ball::bd.test(
          X_part, 
          Y_part, 
          num.permutations = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, "p-value"] <- round(test_result$p.value, 3)
        cat("\n### Fold", i, "(ball) ###\n")
        print(test_result)
        
      } else if (method == "kernel") {
        
        sig <- med_sigma(X_part, Y_part)
        test_result <- kerTests::kertests(
          X_part, 
          Y_part, 
          sigma = sig
        )
        res[[paste0("Fold_", i)]] <- test_result
        results_table[i, "p-value"] <- round(test_result$p.value, 3)
        cat("\n### Fold", i, "(kernel) ###\n")
        print(test_result)
      }
    }
    
  })
  
  writeLines(output_content, con = output_file)
  cat("Saved output to file:", output_file, "\n")
  
  # 테이블 CSV로 저장
  write.csv(as.data.frame(results_table), file = paste0(sub("\\.txt$", ".csv", output_file)), row.names = TRUE)
  cat("Saved table to CSV file:", paste0(sub("\\.txt$", ".csv", output_file)), "\n")
  
  return(res)
}

batch_tests <- function(data_list, source = "agoda", kfds = 10, methods = c("eqdist", "ball"), seed = 7) {
  
  set.seed(seed)
  methods <- match.arg(methods, several.ok = TRUE)
  
  generate_filename <- function(X_name, Y_name, method, seed) {
    paste0(
      "comparison_tests_",
      X_name, "_",
      Y_name, "_",
      method, "_",
      seed
    )
  }
  
  all_combinations <- combn(names(data_list), 2, simplify = FALSE)
  
  for (comb in all_combinations) {
    X_name <- comb[1]
    Y_name <- comb[2]
    X <- data_list[[X_name]]
    Y <- data_list[[Y_name]]
    
    n <- nrow(X)
    if (nrow(X) != nrow(Y)) {
      stop("X와 Y의 행(샘플) 수가 일치해야 합니다.")
    }
    
    folds <- split(sample(1:n), rep(1:kfds, length.out = n))
    res <- list()
    results_table <- matrix("", nrow = kfds, ncol = length(methods), dimnames = list(
      paste0("fold", 1:kfds),
      methods
    ))
    
    output_content <- capture.output({
      cat("==== Comparison Tests ====\n")
      cat("Methods  :", paste(methods, collapse = ", "), "\n")
      cat("X_name   :", X_name, "\n")
      cat("Y_name   :", Y_name, "\n")
      cat("k-fold   :", kfds, "\n")
      cat("Seed     :", seed, "\n")
      cat("Total N  :", n, "\n")
      cat("Source   :", source, "\n\n")
      
      cat("==== Fold Indices ====\n")
      for (i in seq_len(kfds)) {
        cat("Fold", i, ":", paste(folds[[i]], collapse = ", "), "\n")
      }
      cat("\n==== Test Results ====\n")
      
      for (i in seq_len(kfds)) {
        current_idx <- folds[[i]]
        X_part <- X[current_idx, , drop = FALSE]
        Y_part <- Y[current_idx, , drop = FALSE]
        
        for (method in methods) {
          if (method == "eqdist") {
            combined_data <- rbind(X_part, Y_part)
            test_result <- energy::eqdist.etest(
              dist(combined_data), 
              sizes = c(nrow(X_part), nrow(Y_part)), 
              R = 199
            )
            res[[paste0("Fold_", i, "_", method)]] <- test_result
            results_table[i, method] <- round(test_result$p.value, 3)
            cat("\n### Fold", i, "(", method, ") ###\n")
            print(test_result)
          } else if (method == "ball") {
            test_result <- Ball::bd.test(
              X_part, 
              Y_part, 
              num.permutations = 199
            )
            res[[paste0("Fold_", i, "_", method)]] <- test_result
            results_table[i, method] <- round(test_result$p.value, 3)
            cat("\n### Fold", i, "(", method, ") ###\n")
            print(test_result)
          }
        }
      }
    })
    
    output_file <- generate_filename(X_name, Y_name, paste(methods, collapse = "_"), seed)
    writeLines(output_content, con = paste0(output_file, ".txt"))
    cat("Saved output to file:", paste0(output_file, ".txt"), "\n")
    
    write.csv(as.data.frame(results_table), file = paste0(sub("\\.txt$", ".csv", output_file)), row.names = TRUE)
    cat("Saved table to CSV file:", paste0(sub("\\.txt$", ".csv", output_file)), "\n")
  }
}

###############################################################################
'
Load
'
###############################################################################

# quora
load("svd_gen1_quora_0_1.rds")
svd_gen1_quora_0_1 <- svd_quora
load("svd_gen1_quora_0_4.rds")
svd_gen1_quora_0_4 <- svd_quora
load("svd_gen1_quora_0_7.rds")
svd_gen1_quora_0_7 <- svd_quora
load("svd_gen1_quora_1_0.rds")
svd_gen1_quora_1_0 <- svd_quora

svd_gen1_quora_1_5 <- readRDS("gen1_SVD_qr_1_5.RDS")


# cnn
load("svd_gen1_cnn_news_0_1.rds")
svd_gen1_cnn_news_0_1 <- svd_cnn_news

load("svd_gen1_cnn_news_0_4.rds")
svd_gen1_cnn_news_0_4 <- svd_cnn_news

load("svd_gen1_cnn_news_0_7.rds")
svd_gen1_cnn_news_0_7 <- svd_cnn_news

load("svd_gen1_cnn_news_1_0.rds")
svd_gen1_cnn_news_1_0 <- svd_cnn_news

svd_gen1_cnn_news_1_5 <- readRDS("gen1_SVD_cnn_1_5.RDS")


# squad2
load("svd_gen1_squad_2_0_1.rds")
svd_gen1_squad_2_0_1 <- svd_squad_2

load("svd_gen1_squad_2_0_4.rds")
svd_gen1_squad_2_0_4 <- svd_squad_2

load("svd_gen1_squad_2_0_7.rds")
svd_gen1_squad_2_0_7 <- svd_squad_2

load("svd_gen1_squad_2_1_0.rds")
svd_gen1_squad_2_1_0 <- svd_squad_2

svd_gen1_squad_2_1_5 <- readRDS("gen1_SVD_sqd_1_5.RDS")

# agoda

SVD_gen1_0_1 <- readRDS(file = "D:/ago-ebd-svd/SVD_gen1_0_1.rds")
SVD_gen1_0_4 <- readRDS(file = "D:/ago-ebd-svd/SVD_gen1_0_4.rds")
SVD_gen1_0_7 <- readRDS(file = "D:/ago-ebd-svd/SVD_gen1_0_7.rds")
SVD_gen1_1_0 <- readRDS(file = "D:/ago-ebd-svd/SVD_gen1_1_0.rds")
SVD_gen1_1_5 <- readRDS(file = "D:/ago-ebd-svd/SVD_gen1_1_5.rds")



###############################################################################
'
reduction
'
###############################################################################

dim = 2

# quora
red_gen1_quora_0_1 <- svd_gen1_quora_0_1$u %*% rbind(
  diag(svd_gen1_quora_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_0_4 <- svd_gen1_quora_0_4$u %*% rbind(
  diag(svd_gen1_quora_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_0_7 <- svd_gen1_quora_0_7$u %*% rbind(
  diag(svd_gen1_quora_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_1_0 <- svd_gen1_quora_1_0$u %*% rbind(
  diag(svd_gen1_quora_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_1_5 <- svd_gen1_quora_1_5$u %*% rbind(
  diag(svd_gen1_quora_1_5$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# cnn
red_gen1_cnn_news_0_1 <- svd_gen1_cnn_news_0_1$u %*% rbind(
  diag(svd_gen1_cnn_news_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_0_4 <- svd_gen1_cnn_news_0_4$u %*% rbind(
  diag(svd_gen1_cnn_news_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_0_7 <- svd_gen1_cnn_news_0_7$u %*% rbind(
  diag(svd_gen1_cnn_news_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_1_0 <- svd_gen1_cnn_news_1_0$u %*% rbind(
  diag(svd_gen1_cnn_news_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_1_5 <- svd_gen1_cnn_news_1_5$u %*% rbind(
  diag(svd_gen1_cnn_news_1_5$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)


# squad2
red_gen1_squad_2_0_1 <- svd_gen1_squad_2_0_1$u %*% rbind(
  diag(svd_gen1_squad_2_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_0_4 <- svd_gen1_squad_2_0_4$u %*% rbind(
  diag(svd_gen1_squad_2_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_0_7 <- svd_gen1_squad_2_0_7$u %*% rbind(
  diag(svd_gen1_squad_2_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_1_0 <- svd_gen1_squad_2_1_0$u %*% rbind(
  diag(svd_gen1_squad_2_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_1_5 <- svd_gen1_squad_2_1_5$u %*% rbind(
  diag(svd_gen1_squad_2_1_5$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# agoda
red_gen1_0_1 <- SVD_gen1_0_1$u %*% rbind(diag(SVD_gen1_0_1$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_0_4 <- SVD_gen1_0_4$u %*% rbind(diag(SVD_gen1_0_4$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_0_7 <- SVD_gen1_0_7$u %*% rbind(diag(SVD_gen1_0_7$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_0 <- SVD_gen1_1_0$u %*% rbind(diag(SVD_gen1_1_0$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_5 <- SVD_gen1_1_5$u %*% rbind(diag(SVD_gen1_1_5$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))




###############################################################################
'
Testing
'
###############################################################################

setwd("D:/Q2")

# Dataset-specific data lists
quora_data_list <- list(
  "quora_0_1" = red_gen1_quora_0_1,
  "quora_0_4" = red_gen1_quora_0_4,
  "quora_0_7" = red_gen1_quora_0_7,
  "quora_1_0" = red_gen1_quora_1_0,
  "quora_1_5" = red_gen1_quora_1_5
)

cnn_data_list <- list(
  "cnn_0_1" = red_gen1_cnn_news_0_1,
  "cnn_0_4" = red_gen1_cnn_news_0_4,
  "cnn_0_7" = red_gen1_cnn_news_0_7,
  "cnn_1_0" = red_gen1_cnn_news_1_0,
  "cnn_1_5" = red_gen1_cnn_news_1_5
)

squad2_data_list <- list(
  "squad2_0_1" = red_gen1_squad_2_0_1,
  "squad2_0_4" = red_gen1_squad_2_0_4,
  "squad2_0_7" = red_gen1_squad_2_0_7,
  "squad2_1_0" = red_gen1_squad_2_1_0,
  "squad2_1_5" = red_gen1_squad_2_1_5
)

agoda_data_list <- list(
  "agoda_0_1" = red_gen1_0_1,
  "agoda_0_4" = red_gen1_0_4,
  "agoda_0_7" = red_gen1_0_7,
  "agoda_1_0" = red_gen1_1_0,
  "agoda_1_5" = red_gen1_1_5
)

# Perform batch tests for each dataset
batch_tests(
  data_list = quora_data_list,
  source = "quora",
  kfds = 5,
  methods = c("eqdist", "ball"),
  seed = 7
)

batch_tests(
  data_list = cnn_data_list,
  source = "cnn",
  kfds = 5,
  methods = c("eqdist", "ball"),
  seed = 7
)

batch_tests(
  data_list = squad2_data_list,
  source = "squad2",
  kfds = 5,
  methods = c("eqdist", "ball"),
  seed = 7
)

batch_tests(
  data_list = agoda_data_list,
  source = "agoda",
  kfds = 10,
  methods = c("eqdist", "ball"),
  seed = 7
)
















