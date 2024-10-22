new_try = function(dat1, dat2, alpha, itrN = 1000, K=3){
  dat1_cl = kmeans(dat1, K)
  dat2_cl = kmeans(dat2, K)
  
  dat1_dist = sqrt(apply((dat1 - dat1_cl$center[dat1_cl$cluster,])^2, 1, sum))
  
  dat2_dist = sqrt(apply((dat2 - dat2_cl$center[dat2_cl$cluster,])^2, 1, sum))
  
  or_stat = sum(dat1_dist- dat2_dist)
  
  dat1_mirror_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  dat2_mirror_cent = matrix(NA, nrow=K, ncol=ncol(dat2))
  
  dat1_mirror_K_dist = matrix(NA, ncol=K, nrow=nrow(dat1))
  dat2_mirror_K_dist = matrix(NA, ncol=K, nrow=nrow(dat2))
  
  for (k in 1:K){
    dat2_mirror_cent[k,] = apply(dat2[dat1_cl$cluster==k, ],2,mean)
    dat1_mirror_cent[k,] = apply(dat1[dat2_cl$cluster==k, ],2,mean)
    
    # dat1_mirror_K_dist[,k] = sqrt(apply((dat1 - matrix(rep(dat1_mirror_cent[k,], nrow(dat1)), nrow=nrow(dat1), byrow=T))^2,1,sum))
    # dat2_mirror_K_dist[,k] = sqrt(apply((dat2 - matrix(rep(dat2_mirror_cent[k,], nrow(dat2)), nrow=nrow(dat2), byrow=T))^2,1,sum))
  }
  
  
  dat1_mirror_dist = sqrt(apply((dat1 -dat1_mirror_cent[dat2_cl$cluster,])^2, 1, sum))
  dat2_mirror_dist = sqrt(apply((dat2 -dat2_mirror_cent[dat1_cl$cluster,])^2, 1, sum))
  
  # generate itrN number of permutation samples
  stat_vec = rep(NA, itrN)
  
  for (itr in 1:itrN){
    new_dat1 = dat1
    new_dat2 = dat2
    
    TF_ind = sample(c(T,F), size=nrow(dat1), replace=T)
    
    new_dat1_dist = c(dat1_dist[TF_ind==1], dat1_mirror_dist[TF_ind==0])
    new_dat2_dist = c(dat2_dist[TF_ind==1], dat2_mirror_dist[TF_ind==0])
    
    new_stat = sum(new_dat1_dist - new_dat2_dist)
    stat_vec[itr] = new_stat
  }
  
  reject = (quantile(stat_vec, alpha/2) > or_stat) |  (quantile(stat_vec, 1-alpha/2) < or_stat) 
  
  pval = mean(stat_vec >= or_stat)
  
  return(list(stat_vec = stat_vec, or_stat = or_stat, p = pval, p_test = 2*min(pval, 1-pval), reject=reject))
}

plot_clusters <- function(dat, title, lim = c(-5, 5)) {
  
  plot(dat[1:n, ], col = 'red', pch = 16, xlab = "X", ylab = "Y", main = title, cex = 1.2, xlim = lim, ylim = lim)
  points(dat[(n+1):(2*n), ], col = 'blue', pch = 16)
  points(dat[(2*n+1):(3*n), ], col = 'green', pch = 16)
  
  legend("topleft", inset = c(-0.3, 0), legend = c("Cluster 1", "Cluster 2", "Cluster 3"),
         col = c("red", "blue", "green"), pch = 19, cex = 0.8, bty = "n")
}

plot_combined <- function(dat1, dat2, title, lim) {
  
  plot(dat1[1:n, ], col = 'red', pch = 16, xlab = "X", ylab = "Y", main = title, cex = 1.2, xlim = lim, ylim = lim)
  points(dat1[(n+1):(2*n), ], col = 'blue', pch = 16)
  points(dat1[(2*n+1):(3*n), ], col = 'green', pch = 16)
  
  points(dat2[1:n, ], col = 'orange', pch = 17)
  points(dat2[(n+1):(2*n), ], col = 'purple', pch = 17)
  points(dat2[(2*n+1):(3*n), ], col = 'brown', pch = 17)
  
}

get_centroid <- function(cluster) {
  return(colMeans(cluster))
}

scale_cluster <- function(cluster, scale_factor) {
  
  centroid <- get_centroid(cluster)
  cluster_centered <- cluster - matrix(centroid, nrow = nrow(cluster), ncol = ncol(cluster), byrow = TRUE)
  cluster_scaled <- cluster_centered * scale_factor
  cluster_transformed <- cluster_scaled + matrix(centroid, nrow = nrow(cluster), ncol = ncol(cluster), byrow = TRUE)
  
  return(cluster_transformed)
}

gen_donut <- function(n, inner_radius = 0.5, outer_radius = 1.5, centroid = c(0, 0), noise_level = 0.2) {
  theta <- runif(n, 0, 2*pi)
  r <- sqrt(runif(n, inner_radius^2, outer_radius^2))
  x <- r * cos(theta)
  y <- r * sin(theta)
  
  donut_data <- cbind(x, y) + matrix(rep(centroid, n), ncol=2, byrow=TRUE)
  
  noise <- noise_level * matrix(rnorm(n*2), ncol=2)
  donut_data <- donut_data + noise
  
  return(donut_data)
}

set.seed(241016)
library(mvtnorm)

# 1st center; 2nd center; 3rd center
cent1 = c(sqrt(3), 1); cent2 = c(-sqrt(3),1); cent3 = c(0,-2)
# 1st; 2nd, and 3rd sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2)/2,0,0,sqrt(2)), nrow=2)

n=1000

realization = 1000
pvec=NULL

Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
Z3 = rmvnorm(n, mean= cent3, sigma = sig3)

X1 = Z1 + .2 * matrix(rnorm(n*2), ncol=2)
X2 = Z2 + .2 * matrix(rnorm(n*2), ncol=2)
X3 = Z3 + .2 * matrix(rnorm(n*2), ncol=2)

donut_X1 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent1)
donut_X2 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent2)
donut_X3 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent3)

dat1 = rbind(X1, X2, X3)
dat2 = rbind(donut_X1, donut_X2, donut_X3)


plot_clusters(dat2, 'donut')
