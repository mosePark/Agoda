# 필요한 라이브러리 불러오기
library(argparse)
library(ggplot2)

# RStudio에서 실행 시 기본값 설정
if (interactive()) {
  args <- list(sz = 5000, k = 3, iter = 1000, k_near = 250)
} else {
  # Argparse 설정
  parser <- ArgumentParser(description = 'Run permutation test and Rand Index analysis.')
  parser$add_argument('--sz', type = 'integer', default = 5000, help = 'Sample size')
  parser$add_argument('--k', type = 'integer', default = 5, help = 'Number of clusters for kmeans')
  parser$add_argument('--iter', type = 'integer', default = 1000, help = 'Number of iterations for permutation test')
  parser$add_argument('--k_near', type = 'integer', default = 5, help = 'Number of nearest neighbors for distance calculation')
  
  # 인자 파싱
  args <- parser$parse_args()
}


# 파싱된 인자 확인
sz <- args$sz
k <- args$k
iter <- args$iter
k_near <- args$k_near

# 확인 출력 (디버깅용)
cat("Sample size (sz):", sz, "\n")
cat("Number of clusters (k):", k, "\n")
cat("Number of iterations (iter):", iter, "\n")
cat("Number of nearest neighbors (k_near):", k_near, "\n")

# 나머지 코드는 기존과 동일
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

try <- function(dat1, dat2, itrN = 500, K=4, k_near=5){
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

plot_clustered_data <- function(df, cluster, title, file_name) {
  df <- data.frame(X = df[,1], Y = df[,2], Cluster = as.factor(cluster))  # Create dataframe
  
  p <- ggplot(df, aes(x = X, y = Y, color = Cluster)) +
    geom_point(size = 2) +
    labs(title = title, x = "Dimension 1", y = "Dimension 2") +
    theme_minimal(base_size = 15)
  
  print(p)
  
  ggsave(file_name, plot = p, dpi = 1000, width = 8, height = 6)
}

plot_pvalue_distribution_ggplot <- function(ri1, ri2, rf, title) {
  # Calculate (rand1 + ri2) / 2
  ri_avg <- (ri1 + ri2) * 0.5
  df <- data.frame(ri_avg)  # Create a dataframe for ggplot
  
  # Create the ggplot histogram
  p <- ggplot(df, aes(x = ri_avg)) +
    geom_histogram(binwidth = 0.02, fill = "lightblue", color = "black", alpha = 0.7) +
    geom_vline(xintercept = rf, color = "red", linetype = "dashed", linewidth = 1.2) +  # Fixed: replaced size with linewidth
    labs(title = title, x = "Mean RI (Random Index Mean)", y = "Frequency") +
    theme_minimal(base_size = 15)  # Use a minimal theme for simplicity
  
  # Display the plot
  print(p)
  
  # Calculate p-value
  pvalue <- mean(ri_avg < rf)
  cat("p-value:", pvalue, "\n")
  
  # Save the plot with high DPI (1000) using ggsave
  ggsave(paste0(title, "_distribution.png"), plot = p, dpi = 1000, width = 8, height = 6)
}


# 데이터 로드 및 샘플링
set.seed(241003)
setwd("D:/mose/data/ablation2")

df_4 <- read.csv("df_4.csv", header=T)
df_5 <- read.csv("df_5.csv", header=T)
df_6 <- read.csv("df_6.csv", header=T)

sample_ind = sample(1:nrow(df_4), size=sz)

or_emb = make_matrix(df_6$ori.ebd[sample_ind])
gen1_emb = make_matrix(df_4$X0.7.gen1.ebd[sample_ind])
gen2_emb = make_matrix(df_5$X0.7.gen2.ebd[sample_ind])
gen1_1_emb = make_matrix(df_6$X0.7.gen1.1.ebd[sample_ind])

#############################
######### EDA ###############
#############################

## SVD 및 차원 축소 (2차원)
SVD_or = svd(or_emb)
SVD_gen1 = svd(gen1_emb)
SVD_gen2 = svd(gen2_emb)
SVD_gen1_1 = svd(gen1_1_emb)

red_or = or_emb %*% SVD_or$v[,1:2]
red_gen1 = gen1_emb %*% SVD_gen1$v[,1:2]
red_gen2 = gen2_emb %*% SVD_gen2$v[,1:2]
red_gen1_1 = gen1_1_emb %*% SVD_gen1_1$v[,1:2]

## 클러스터링 및 시각화
or_clust = kmeans(red_or, k)
gen1_clust = kmeans(red_gen1, k)
gen2_clust = kmeans(red_gen2, k)
gen1_1_clust = kmeans(red_gen1_1, k)

setwd("C:/Users/mose/Desktop/simulation")

plot_clustered_data(red_or, gen1_clust$cluster, "compare the results across datasets", "red_or_gen1_clusters.png")
plot_clustered_data(red_gen1, gen1_clust$cluster, "compare the results across datasets", "red_gen1_gen1_clusters.png")
plot_clustered_data(red_gen2, gen1_clust$cluster, "compare the results across datasets", "red_gen2_gen1_clusters.png")
plot_clustered_data(red_gen1_1, gen1_clust$cluster, "compare the results across datasets", "red_gen1_1_gen1_clusters.png")

rand.index(gen1_clust$cluster, gen2_clust$cluster)
rand.index(gen1_clust$cluster, or_clust$cluster)
rand.index(gen1_clust$cluster, gen1_1_clust$cluster)
rand.index(gen2_clust$cluster, gen1_1_clust$cluster)

library(mclust)

adjustedRandIndex(gen1_clust$cluster, gen2_clust$cluster)
adjustedRandIndex(gen1_clust$cluster, or_clust$cluster)
adjustedRandIndex(gen1_clust$cluster, gen1_1_clust$cluster)
adjustedRandIndex(gen2_clust$cluster, gen1_1_clust$cluster)

#####################################################
## 순열검정 및 p-value 계산 (15차원으로 차원축소) ##
#####################################################

# SVD를 사용하여 15차원으로 차원 축소
red_or_15 = or_emb %*% SVD_or$v[,1:15]
red_gen1_15 = gen1_emb %*% SVD_gen1$v[,1:15]
red_gen2_15 = gen2_emb %*% SVD_gen2$v[,1:15]
red_gen1_1_15 = gen1_1_emb %*% SVD_gen1_1$v[,1:15]


# Rand Index와 순열검정
g1g2 = try(red_gen1_15, red_gen2_15, itrN = iter, K=k, k_near=k_near)
org1 = try(red_gen1_15, red_or_15, itrN = iter, K=k, k_near=k_near)
org2 = try(red_gen2_15, red_or_15, itrN = iter, K=k, k_near=k_near)
g1g11 = try(red_gen1_15, red_gen1_1_15, itrN = iter, K=k, k_near=k_near)
g2g11 = try(red_gen2_15, red_gen1_1_15, itrN = iter, K=k, k_near=k_near)
org11 = try(red_or_15, red_gen1_1_15, itrN = iter, K=k, k_near=k_near)

# p-value 계산
cat("p-value g1g2:", mean((g1g2$ri2 + g1g2$ri1) * 0.5 < g1g2$rf), "\n")
cat("p-value org1:", mean((org1$ri2 + org1$ri1) * 0.5 < org1$rf), "\n")
cat("p-value org2:", mean((org2$ri2 + org2$ri1) * 0.5 < org2$rf), "\n")
cat("p-value g1g11:", mean((g1g11$ri2 + g1g11$ri1) * 0.5 < g1g11$rf), "\n")
cat("p-value g2g11:", mean((g2g11$ri2 + g2g11$ri1) * 0.5 < g2g11$rf), "\n")
cat("p-value org11:", mean((org11$ri2 + org11$ri1) * 0.5 < org11$rf), "\n")

# p-value dist
# g1g2
plot_pvalue_distribution_ggplot(g1g2$ri1, g1g2$ri2, g1g2$rf, "g1g2")
# org1
plot_pvalue_distribution_ggplot(org1$ri1, org1$ri2, org1$rf, "org1")
# org2
plot_pvalue_distribution_ggplot(org2$ri1, org2$ri2, org2$rf, "org2")
# g1g11
plot_pvalue_distribution_ggplot(g1g11$ri1, g1g11$ri2, g1g11$rf, "g1g11")
# g2g11
plot_pvalue_distribution_ggplot(g2g11$ri1, g2g11$ri2, g2g11$rf, "g2g11")
# org11
plot_pvalue_distribution_ggplot(org11$ri1, org11$ri2, org11$rf, "org2")
