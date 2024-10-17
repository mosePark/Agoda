# CRAN mirror setting
options(repos = c(CRAN = 'https://cran.r-project.org'))

# packages

if (!requireNamespace("progress", quietly = TRUE)) {
  install.packages("progress", lib = "/home1/mose1103/R/library")
}


# folder setting
base_path = "/home1/mose1103/agoda/simulation/results"
simulation_id <- format(Sys.time(), "%H시 %M분 %S초_Real_0_7")
save_path <- file.path(base_path, simulation_id)

if (!dir.exists(save_path)) {
  dir.create(save_path, recursive = TRUE)
}

### df to matrix
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


### test function is modified.
new_try = function(dat1, dat2, alpha, itrN = 1000, K=3){
  dat1_cl = kmeans(dat1, K)
  dat2_cl = kmeans(dat2, K)
  
  dat1_dist = sqrt(apply((dat1 -dat1_cl$center[dat1_cl$cluster,])^2, 1, sum))

  dat2_dist = sqrt(apply((dat2 -dat2_cl$center[dat2_cl$cluster,])^2, 1, sum))
  
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


# Load libraries

library(progress)

# Setting seed
set.seed(140413)

'
Load Data
'

df_4 <- read.csv("/home1/mose1103/agoda/simulation/data/df_4.csv", header=T)
df_5 <- read.csv("/home1/mose1103/agoda/simulation/data/df_5.csv", header=T)
df_6 <- read.csv("/home1/mose1103/agoda/simulation/data/df_6.csv", header=T)

sz = 5000
iter = 100000
k = 3
realization = 30000 / sz


'
Testing
'

pb <- progress_bar$new(
  format = "  진행중 [:bar] :percent 완료 - 경과시간 :elapsed - 예상완료시간 :eta",
  total = realization, clear = FALSE, width = 60
)

# stat_vec = matrix(NA, nrow = realization, ncol = 6)
or_stat = matrix(NA, nrow = realization, ncol = 6)
p = matrix(NA, nrow = realization, ncol = 6)
p_test = matrix(NA, nrow = realization, ncol = 6)
reject = matrix(NA, nrow = realization, ncol = 6)

# colnames(stat_vec)
colnames(or_stat) = c("g1g2", "org1", "org2", "g1g11", "g2g11", "org11")
colnames(p) = c("g1g2", "org1", "org2", "g1g11", "g2g11", "org11")
colnames(p_test) = c("g1g2", "org1", "org2", "g1g11", "g2g11", "org11")
colnames(reject) = c("g1g2", "org1", "org2", "g1g11", "g2g11", "org11")

for (i in 1:realization){
  sample_ind = sample(1:nrow(df_4), size=sz)

  or_emb = make_matrix(df_6$ori.ebd[sample_ind])
  gen1_emb = make_matrix(df_4$X0.7.gen1.ebd[sample_ind])
  gen2_emb = make_matrix(df_5$X0.7.gen2.ebd[sample_ind])
  gen1_1_emb = make_matrix(df_6$X0.7.gen1.1.ebd[sample_ind])

  SVD_or = svd(or_emb)
  SVD_gen1 = svd(gen1_emb)
  SVD_gen2 = svd(gen2_emb)
  SVD_gen1_1 = svd(gen1_1_emb)

  red_or = or_emb%*%SVD_or$v[,1:15]
  red_gen1 = gen1_emb%*%SVD_gen1$v[,1:15]
  red_gen2 = gen2_emb%*%SVD_gen2$v[,1:15]
  red_gen1_1 = gen1_1_emb%*%SVD_gen1_1$v[,1:15]

  g1g2 = new_try(red_gen1, red_gen2, alpha=.05, K=k, itrN=iter)
  org1 = new_try(red_or, red_gen1, alpha=.05, K=k, itrN=iter)
  org2 = new_try(red_or, red_gen2, alpha=.05, K=k, itrN=iter)
  g1g11 = new_try(red_gen1, red_gen1_1, alpha=.05, K=k, itrN=iter)
  g2g11 = new_try(red_gen2, red_gen1_1, alpha=.05, K=k, itrN=iter)
  org11 = new_try(red_or, red_gen1_1, alpha=.05, K=k, itrN=iter)

  # stat_vec[i, ] = c(g1g2$stat_vec, org1$stat_vec, org2$stat_vec, g1g11$stat_vec, g2g11$stat_vec, org11$stat_vec)
  or_stat[i, ] = c(g1g2$or_stat, org1$or_stat, org2$or_stat, g1g11$or_stat, g2g11$or_stat, org11$or_stat)
  p[i, ] = c(g1g2$p, org1$p, org2$p, g1g11$p, g2g11$p, org11$p)
  p_test[i, ] = c(g1g2$p_test, org1$p_test, org2$p_test, g1g11$p_test, g2g11$p_test, org11$p_test)
  reject[i, ] = c(g1g2$reject, org1$reject, org2$reject, g1g11$reject, g2g11$reject, org11$reject)
  
  
  if (!pb$finished) {
    pb$tick()
    }
}

# write.csv(res_table$stat_vec, file = file.path(save_path, "stat_vec.csv"), row.names = FALSE)
write.csv(res_table$or_stat, file = file.path(save_path, "or_stat.csv"), row.names = FALSE)
write.csv(res_table$p, file = file.path(save_path, "p_values.csv"), row.names = FALSE)
write.csv(res_table$p_test, file = file.path(save_path, "p_test.csv"), row.names = FALSE)
write.csv(res_table$reject, file = file.path(save_path, "reject.csv"), row.names = FALSE)
