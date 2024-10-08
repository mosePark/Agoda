# 시뮬레이션 결과를 저장할 폴더 이름 설정

base_path = "C:/Users/simulation/results"
simulation_id <- format(Sys.time(), "%H시 %M분 %S초") # 날짜 기반 폴더 이름 생성
save_path <- file.path(base_path, simulation_id)

# 폴더가 없으면 생성
if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}


# Function
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
  
  # calculate pairwise distance for both datasets
  dat1_dist = as.matrix(dist(dat1))
  dat2_dist = as.matrix(dist(dat2))
  
  # calculate the total distance for an observation  
  tot_dist1 = apply(dat1_dist,2,sum)
  tot_dist2 = apply(dat2_dist,2,sum)
  
  diag(dat1_dist) = NA
  diag(dat2_dist) = NA
  
  # a function that outputs the indices of k_near (number) closest neighbors
  k_near_take = function(x, k_near){
    order(x)[1:k_near]
  }
  
  dat1_near_mat = apply(dat1_dist, 2, k_near_take, k_near)
  dat2_near_mat = apply(dat2_dist, 2, k_near_take, k_near)
  
  rand1 = rep(NA, itrN)
  rand2 = rep(NA, itrN)
  
  # generate itrN number of permutation samples
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

# Load libraries
library(MASS)
library(ggplot2)
library(progress)

# Setting seed
set.seed(241007)

# Define centroids (mean vectors)
mu1 <- c(2, 2)
mu2 <- c(-2, 2)
mu3 <- c(0, -3)

# Adjusted covariance matrices for more overlap
cov1 <- matrix(c(1.5, 0.6, 0.6, 1.5), nrow = 2)
cov2 <- matrix(c(1.5, -0.4, -0.4, 1.5), nrow = 2)
cov3 <- matrix(c(1.5, 0.3, 0.3, 1.5), nrow = 2)

# Sampling 100 data points ~ MultiNormal dist for each centroid
cluster1 <- mvrnorm(100, mu = mu1, Sigma = cov1)
cluster2 <- mvrnorm(100, mu = mu2, Sigma = cov2)
cluster3 <- mvrnorm(100, mu = mu3, Sigma = cov3)

# Combine the data into a single data frame for ggplot
dat1 <- data.frame(
  X1 = c(cluster1[,1], cluster2[,1], cluster3[,1]),
  X2 = c(cluster1[,2], cluster2[,2], cluster3[,2]),
  Cluster = factor(rep(1:3, each = 100))
)

# Define colors for clusters
cluster_colors <- c("orange", "green", "blue")

# Create the ggplot
p <- ggplot(dat1, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Multivariate Normal Distribution with Cluster Overlap") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-6, 6))

# Display plot
p

ggsave(file.path(save_path, "dat1_cluster_plot.png"), plot = p, width = 8, height = 6, dpi = 1000)


# Define linear transformation parameters for each cluster
a1 <- 1.5; b1 <- c(1, -2)
a2 <- 1.3; b2 <- c(-3, 4)
a3 <- 0.2; b3 <- c(5, -2)

# saving value setting
save_path_txt <- file.path(save_path, "simulation_params.txt")

# 파일에 변수 저장
sink(save_path_txt)  # 출력 방향을 텍스트 파일로 설정
cat("mu1:\n"); print(mu1)
cat("\nmu2:\n"); print(mu2)
cat("\nmu3:\n"); print(mu3)
cat("\ncov1:\n"); print(cov1)
cat("\ncov2:\n"); print(cov2)
cat("\ncov3:\n"); print(cov3)
cat("\na1:\n"); print(a1)
cat("\nb1:\n"); print(b1)
cat("\na2:\n"); print(a2)
cat("\nb2:\n"); print(b2)
cat("\na3:\n"); print(a3)
cat("\nb3:\n"); print(b3)
sink()  # 출력 방향을 다시 콘솔로 변경

# Apply linear transformation to each cluster
cluster1_transformed <- a1 * cluster1 + matrix(rep(b1, each = 100), ncol = 2, byrow = TRUE)
cluster2_transformed <- a2 * cluster2 + matrix(rep(b2, each = 100), ncol = 2, byrow = TRUE)
cluster3_transformed <- a3 * cluster3 + matrix(rep(b3, each = 100), ncol = 2, byrow = TRUE)

# Combine transformed data into a new data frame for plotting
dat2 <- data.frame(
  X1 = c(cluster1_transformed[,1], cluster2_transformed[,1], cluster3_transformed[,1]),
  X2 = c(cluster1_transformed[,2], cluster2_transformed[,2], cluster3_transformed[,2]),
  Cluster = factor(rep(1:3, each = 100))
)

# Plot the transformed data
p2 <- ggplot(dat2, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Transformed Data with Linear Transformation (data2)") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-10, 10), ylim = c(-10, 10))

# Display plot
p2

ggsave(file.path(save_path, "dat2_transformed_plot.png"), plot = p2, width = 8, height = 6, dpi = 1000)


# Simulation steps continue...

m1 = as.matrix(dat1[, c("X1", "X2")])
m2 = as.matrix(dat2[, c("X1", "X2")])

N <- 100  # Set this to your desired number of iterations

# Initialize a vector to store the results from each run
distribution_results <- numeric(N)

# Create a progress bar
pb <- progress_bar$new(
  format = "  진행중 [:bar] :percent 완료 - 경과시간 :elapsed - 예상완료시간 :eta",
  total = N, clear = FALSE, width = 60
)

# Run the process N times with a progress bar
for (i in 1:N) {
  output <- try(m1, m2, itrN = 1000, K = 3, k_near = 10)
  distribution_results[i] <- mean((output$ri2 + output$ri1) * 0.5 < output$rf)
  
  # Update the progress bar
  if (!pb$finished) {
    pb$tick()
  }
}

# Plot the distribution of the p-value results
hist_plot <- hist(distribution_results, breaks = 20, col = "skyblue", 
                  main = "Distribution of p-value",
                  xlab = "Mean value", ylab = "Frequency")

# Save the histogram plot with high DPI
ggsave(file.path(save_path, "p_value_distribution_plot.png"), plot = hist_plot, width = 8, height = 6, dpi = 1000)





