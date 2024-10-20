### Hierarchical Cl

new_try_hc = function(dat1, dat2, alpha, itrN = 1000, K=3){
  
  dat1_cl = cutree(hclust(dist(dat1), method = "ward.D2"), k = K)
  dat2_cl = cutree(hclust(dist(dat2), method = "ward.D2"), k = K)
  
  dat1_centers = sapply(1:K, function(k) colMeans(dat1[dat1_cl == k, , drop = FALSE]))
  dat2_centers = sapply(1:K, function(k) colMeans(dat2[dat2_cl == k, , drop = FALSE]))
  
  dat1_dist = sqrt(apply((dat1 - t(dat1_centers[, dat1_cl]))^2, 1, sum))
  dat2_dist = sqrt(apply((dat2 - t(dat2_centers[, dat2_cl]))^2, 1, sum))
  
  or_stat = sum(dat1_dist - dat2_dist)
  
  dat1_mirror_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  dat2_mirror_cent = matrix(NA, nrow=K, ncol=ncol(dat2))
  
  dat1_mirror_K_dist = matrix(NA, ncol=K, nrow=nrow(dat1))
  dat2_mirror_K_dist = matrix(NA, ncol=K, nrow=nrow(dat2))
  
  for (k in 1:K){
    dat2_mirror_cent[k,] = apply(dat2[dat1_cl == k, ], 2, mean)
    dat1_mirror_cent[k,] = apply(dat1[dat2_cl == k, ], 2, mean)
    
    # dat1_mirror_K_dist[,k] = sqrt(apply((dat1 - matrix(rep(dat1_mirror_cent[k,], nrow(dat1)), nrow=nrow(dat1), byrow=T))^2,1,sum))
    # dat2_mirror_K_dist[,k] = sqrt(apply((dat2 - matrix(rep(dat2_mirror_cent[k,], nrow(dat2)), nrow=nrow(dat2), byrow=T))^2,1,sum))
  }
  
  dat1_mirror_dist = sqrt(apply((dat1 - dat1_mirror_cent[dat2_cl,])^2, 1, sum))
  dat2_mirror_dist = sqrt(apply((dat2 - dat2_mirror_cent[dat1_cl,])^2, 1, sum))
  
  stat_vec = rep(NA, itrN)
  
  for (itr in 1:itrN){
    new_dat1 = dat1
    new_dat2 = dat2
    
    TF_ind = sample(c(TRUE, FALSE), size = nrow(dat1), replace = TRUE)
    
    new_dat1_dist = c(dat1_dist[TF_ind], dat1_mirror_dist[!TF_ind])
    new_dat2_dist = c(dat2_dist[TF_ind], dat2_mirror_dist[!TF_ind])
    
    new_stat = sum(new_dat1_dist - new_dat2_dist)
    stat_vec[itr] = new_stat
  }
  
  reject = (quantile(stat_vec, alpha / 2) > or_stat) | (quantile(stat_vec, 1 - alpha / 2) < or_stat)
  
  pval = mean(stat_vec >= or_stat)
  
  return(list(stat_vec = stat_vec, or_stat = or_stat, p = pval, p_test = 2 * min(pval, 1 - pval), reject = reject))
}
