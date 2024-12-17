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
