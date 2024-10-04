library(ggplot2)

# Define function
################################################################################
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
################################################################################
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
################################################################################
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
  '작은 값의 index를 반환하는 함수'
  k_near_take = function(x, k_near){
    order(x)[1:k_near]
  }
  
  dat1_near_mat = apply(dat1_dist, 2, k_near_take, k_near)
  dat2_near_mat = apply(dat2_dist, 2, k_near_take, k_near)
  
  
  rand1 = rep(NA, itrN)
  rand2 = rep(NA, itrN)
  
  # genearte itrN number of permutation samples
  for (itr in 1:itrN){
    # new permuted samples
    new_dat1 = dat1
    new_dat2 = dat2
    
    TF_ind = sample(c(T,F), size=N, replace=T)
    
    for (i in 1:N){
      if (TF_ind[i]){
        
        # calucluate the weights
        mix_coef1 = dat1_dist[i,dat1_near_mat[,i]]
        mix_coef1 = mix_coef1/sum(mix_coef1)
        mix_coef2 = dat2_dist[i,dat2_near_mat[,i]]
        mix_coef2 = mix_coef2/sum(mix_coef2)
        
        # calculate the weighted average
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
################################################################################
plot_clustered_data <- function(df, cluster, title, file_name) {
  df <- data.frame(X = df[,1], Y = df[,2], Cluster = as.factor(cluster))  # Create dataframe
  
  # Create the ggplot scatter plot
  p <- ggplot(df, aes(x = X, y = Y, color = Cluster)) +
    geom_point(size = 2) +  # Size of the points
    labs(title = title, x = "Dimension 1", y = "Dimension 2") +
    theme_minimal(base_size = 15)  # Minimal theme for simplicity
  
  # Display the plot
  print(p)
  
  # Save the plot with high DPI (1000) using ggsave
  ggsave(file_name, plot = p, dpi = 1000, width = 8, height = 6)
}
################################################################################


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



'
코드 시작
'


set.seed(241003)

setwd("D:/mose/data/ablation2")
getwd()

df_4 <- read.csv("df_4.csv", header=T)
df_5 <- read.csv("df_5.csv", header=T)
df_6 <- read.csv("df_6.csv", header=T)


sample_ind = sample(1:nrow(df_4), size=5000)

or_emb = make_matrix(df_6$ori.ebd[sample_ind])
gen1_emb = make_matrix(df_4$X0.7.gen1.ebd[sample_ind])
gen2_emb = make_matrix(df_5$X0.7.gen2.ebd[sample_ind])
gen1_1_emb = make_matrix(df_6$X0.7.gen1.1.ebd[sample_ind])


#############################
#########EDA###############
#############################

## for better visualization, we reduce the dimension to two.
SVD_or = svd(or_emb)
SVD_gen1 = svd(gen1_emb)
SVD_gen2 = svd(gen2_emb)
SVD_gen1_1 = svd(gen1_1_emb)

red_or = or_emb%*%SVD_or$v[,1:2]
red_gen1 = gen1_emb%*%SVD_gen1$v[,1:2]
red_gen2 = gen2_emb%*%SVD_gen2$v[,1:2]
red_gen1_1 = gen1_1_emb%*%SVD_gen1_1$v[,1:2]

## perform k_means and compare the results across datasets.
K = 5
or_clust = kmeans(red_or, K)
gen1_clust = kmeans(red_gen1, K)
gen2_clust = kmeans(red_gen2, K)
gen1_1_clust = kmeans(red_gen1_1, K)

# plot(red_or[,1], red_or[,2], col=gen1_clust$cluster)
# plot(red_gen1[,1], red_gen1[,2], col=gen1_clust$cluster)
# plot(red_gen2[,1], red_gen2[,2], col=gen1_clust$cluster)
# plot(red_gen1_1[,1], red_gen1_1[,2], col=gen1_clust$cluster)

plot_clustered_data(red_or, gen1_clust$cluster, "red_or with gen1 clusters", "red_or_gen1_clusters.png")
plot_clustered_data(red_gen1, gen1_clust$cluster, "red_gen1 with gen1 clusters", "red_gen1_gen1_clusters.png")
plot_clustered_data(red_gen2, gen1_clust$cluster, "red_gen2 with gen1 clusters", "red_gen2_gen1_clusters.png")
plot_clustered_data(red_gen1_1, gen1_clust$cluster, "red_gen1_1 with gen1 clusters", "red_gen1_1_gen1_clusters.png")





rand.index(gen1_clust$cluster, gen2_clust$cluster)
rand.index(gen1_clust$cluster, or_clust$cluster)
rand.index(gen1_clust$cluster, gen1_1_clust$cluster)
rand.index(gen2_clust$cluster, gen1_1_clust$cluster)

## 우리는 gen1-gen2 사이의 클러스터링 결과가 gen1-or 사이의 클러스터링 결과보다 훨씬 더 유사하다는 것을 관찰합니다.

###################################################
## RI_test: perform hypothesis testing for

red_or = or_emb%*%SVD_or$v[,1:15]
red_gen1 = gen1_emb%*%SVD_gen1$v[,1:15]
red_gen2 = gen2_emb%*%SVD_gen2$v[,1:15]
red_gen1_1 = gen1_1_emb%*%SVD_gen1_1$v[,1:15]


dat1 = red_gen1
dat2 = red_or
dat3 = red_gen2
dat4 = red_gen1_1

# testing
g1g2 = try(red_gen1, red_gen2, itrN = 1000, K=5, k_near=5)
org1 = try(red_gen1, red_or, itrN = 1000, K=5, k_near=5)
org2 = try(red_gen2, red_or, itrN = 1000, K=5, k_near=5)
g1g11 = try(red_gen1, red_gen1_1, itrN = 1000, K=5, k_near=5)
g2g11 = try(red_gen2, red_gen1_1, itrN = 1000, K=5, k_near=5)


# my p-value
mean((g1g2$ri2+g1g2$ri1)*.5 < g1g2$rf) # 0.018
mean((org1$ri2+org1$ri1)*.5 < org1$rf) # 0.006
mean((org2$ri2+org2$ri1)*.5 < org2$rf) # 0.002
mean((g1g11$ri2+g1g11$ri1)*.5 < g1g11$rf) # 0.143
mean((g2g11$ri2+g2g11$ri1)*.5 < g2g11$rf) # 0.011



setwd("C:/Users/mose/Desktop/simulation")

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


setwd("C:/Users/mose/Desktop/simulation")
