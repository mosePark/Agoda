base_path = "C:/Users/mose/Desktop/simulation/results"
simulation_id <- format(Sys.time(), "%H시 %M분 %S초") # 날짜 기반 폴더 이름 생성
save_path <- file.path(base_path, simulation_id)


if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}



rand.index <- function(true_labels, predicted_labels) {
  if (length(true_labels) != length(predicted_labels)) {
    stop("The length of the two label vectors must be the same.")
  }
  
  contingency_table <- table(true_labels, predicted_labels)
  
  a <- sum(choose(contingency_table, 2))
  b <- sum(choose(rowSums(contingency_table), 2)) - a
  c <- sum(choose(colSums(contingency_table), 2)) - a
  d <- choose(sum(contingency_table), 2) - (a + b + c)
  
  rand_index <- (a + d) / (a + b + c + d)
  return(rand_index)
}

try = function(dat1, dat2, itrN = 500, K=4, k_near=5){
  dat1_cl = kmeans(dat1, K)$cluster
  dat2_cl = kmeans(dat2, K)$cluster
  
  N = nrow(dat1)
  

  dat1_dist = as.matrix(dist(dat1))
  dat2_dist = as.matrix(dist(dat2))
  

  tot_dist1 = apply(dat1_dist,2,sum)
  tot_dist2 = apply(dat2_dist,2,sum)
  
  diag(dat1_dist) = NA
  diag(dat2_dist) = NA
  

  k_near_take = function(x, k_near){
    order(x)[1:k_near]
  }
  
  dat1_near_mat = apply(dat1_dist, 2, k_near_take, k_near)
  dat2_near_mat = apply(dat2_dist, 2, k_near_take, k_near)
  
  rand1 = rep(NA, itrN)
  rand2 = rep(NA, itrN)
  
 
  for (itr in 1:itrN){
    new_dat1 = dat1
    new_dat2 = dat2
    
    TF_ind = sample(c(T,F), size=N, replace=T)
    
    for (i in 1:N){
      if (TF_ind[i]){
        
        mix_coef1 = dat1_dist[i,dat1_near_mat[,i]]
        mix_coef1 = mix_coef1/sum(mix_coef1)
        mix_coef2 = dat2_dist[i,dat2_near_mat[,i]]
        mix_coef2 = mix_coef2/sum(mix_coef2)
        
        new_dat2_point = apply(mix_coef1*dat2[dat1_near_mat[,i],],2,sum) 
        new_dat1_point = apply(mix_coef2*dat1[dat2_near_mat[,i],],2,sum) 
        
        new_dat1[i,] = new_dat1_point
        new_dat2[i,] = new_dat2_point
        
      } 
    }
    
    new_cl_dat1 = kmeans(new_dat1, K)
    new_cl_dat2 = kmeans(new_dat2, K)
    
    rand1[itr] = rand.index(new_cl_dat1$cluster, dat2_cl)
    rand2[itr] = rand.index(new_cl_dat2$cluster, dat1_cl)
  }
  
  return(list(ri1= rand1, ri2=rand2, rf= rand.index(dat2_cl, dat1_cl)))
}


library(MASS)
library(ggplot2)
library(gridExtra)
library(progress)


set.seed(241007)


mu1 <- c(2, 2)
mu2 <- c(-2, 2)
mu3 <- c(0, -3)


cov1 <- matrix(c(1.5, 0.6, 0.6, 1.5), nrow = 2)
cov2 <- matrix(c(1.5, -0.4, -0.4, 1.5), nrow = 2)
cov3 <- matrix(c(1.5, 0.3, 0.3, 1.5), nrow = 2)


cluster1 <- mvrnorm(100, mu = mu1, Sigma = cov1)
cluster2 <- mvrnorm(100, mu = mu2, Sigma = cov2)
cluster3 <- mvrnorm(100, mu = mu3, Sigma = cov3)


dat1 <- data.frame(
  X1 = c(cluster1[,1], cluster2[,1], cluster3[,1]),
  X2 = c(cluster1[,2], cluster2[,2], cluster3[,2]),
  Cluster = factor(rep(1:3, each = 100))
)


cluster_colors <- c("orange", "green", "blue")


p <- ggplot(dat1, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
 
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Original data") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-6, 6))

p

ggsave(file.path(save_path, "dat1_cluster_plot.png"), plot = p, width = 8, height = 6, dpi = 800)



center1 <- colMeans(cluster1)
center2 <- colMeans(cluster2)
center3 <- colMeans(cluster3)


scaled_cluster1 <- sapply(1:nrow(cluster1), function(i) {
  dist <- cluster1[i,] - center1
  scale_factor <- 0.5 + (1 - 0.5) * (1 - (sqrt(sum(dist^2)) / max(sqrt(rowSums((cluster1 - center1)^2)))))
  center1 + dist * scale_factor
})
scaled_cluster1 <- t(scaled_cluster1)

scaled_cluster2 <- sapply(1:nrow(cluster2), function(i) {
  dist <- cluster2[i,] - center2
  scale_factor <- 0.5 + (1 - 0.5) * (1 - (sqrt(sum(dist^2)) / max(sqrt(rowSums((cluster2 - center2)^2)))))
  center2 + dist * scale_factor
})
scaled_cluster2 <- t(scaled_cluster2)

scaled_cluster3 <- sapply(1:nrow(cluster3), function(i) {
  dist <- cluster3[i,] - center3
  scale_factor <- 0.5 + (1 - 0.5) * (1 - (sqrt(sum(dist^2)) / max(sqrt(rowSums((cluster3 - center3)^2)))))
  center3 + dist * scale_factor
})
scaled_cluster3 <- t(scaled_cluster3)


dat_trans <- data.frame(
  X1 = c(scaled_cluster1[,1], scaled_cluster2[,1], scaled_cluster3[,1]),
  X2 = c(scaled_cluster1[,2], scaled_cluster2[,2], scaled_cluster3[,2]),
  Cluster = factor(rep(1:3, each = 100))
)


p2 <- ggplot(dat_trans, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Transformed Clusters with Distance-Proportional Scaling") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-6, 6))

p2

grid.arrange(p, p2, ncol = 2)

ggsave(file.path(save_path, "dat2_transformed_plot.png"), plot = p2, width = 8, height = 6, dpi = 800)


m1 = as.matrix(dat1[, c("X1", "X2")])
m2 = as.matrix(dat_trans[, c("X1", "X2")])

N <- 100


distribution_results <- numeric(N)

pb <- progress_bar$new(
  format = "  진행중 [:bar] :percent 완료 - 경과시간 :elapsed - 예상완료시간 :eta",
  total = N, clear = FALSE, width = 60
)

iter = 100
k = 3
nbd = 10

for (i in 1:N) {
  output <- try(m1, m2, itrN = iter, K = k, k_near = nbd)
  distribution_results[i] <- mean((output$ri2 + output$ri1) * 0.5 < output$rf)
  
  if (!pb$finished) {
    pb$tick()
  }
}

save_path_txt <- file.path(save_path, "simulation_params.txt")


sink(save_path_txt)
cat("mu1:\n"); print(mu1)
cat("\nmu2:\n"); print(mu2)
cat("\nmu3:\n"); print(mu3)
cat("\ncov1:\n"); print(cov1)
cat("\ncov2:\n"); print(cov2)
cat("\ncov3:\n"); print(cov3)
# cat("\nA1:\n"); print(A1)
# cat("\nb1:\n"); print(b1)
# cat("\nA2:\n"); print(A2)
# cat("\nb2:\n"); print(b2)
# cat("\nA3:\n"); print(A3)
# cat("\nb3:\n"); print(b3)
cat("\nitrN:\n"); print(iter)
cat("\nK:\n"); print(k)
cat("\nk_near:\n"); print(nbd)
cat("\nN:\n"); print(N)
sink()

hist_plot <- ggplot(data.frame(distribution_results), aes(x = distribution_results)) +
  geom_histogram(bins = 20, fill = "skyblue", color = "black") +
  labs(title = "Distribution of p-value",
       x = "Mean value",
       y = "Frequency") +
  theme_minimal(base_size = 15) +
  NULL

hist_plot

ggsave(file.path(save_path, "p_value_distribution_plot.png"), plot = hist_plot, width = 8, height = 6, dpi = 800)
