new_try_2 = function(raw_dat1, raw_dat2, raw_dat3, alpha, K=3, repN=10, standardize=T){
  # standardize in a columnwise manner
  if (standardize){
    dat1 = scale(raw_dat1)
    dat2 = scale(raw_dat2)
    dat3 = scale(raw_dat3)
  }else{
    dat1 = raw_dat1
    dat2 = raw_dat2
    dat3 = raw_dat3
  }
  
  dat2_cl_list = list()
  dat3_cl_list = list()
  dat2_cl_totwithin = rep(NA, repN)
  dat3_cl_totwithin = rep(NA, repN)
  
  for (i in 1:repN){
    dat2_cl_list[[i]] = kmeans(dat2, K)
    dat3_cl_list[[i]] = kmeans(dat3, K)
    
    dat2_cl_totwithin[i] = dat2_cl_list[[i]]$tot.withinss
    dat3_cl_totwithin[i] = dat3_cl_list[[i]]$tot.withinss
  }
  dat2_cl = dat2_cl_list[[which.min(dat2_cl_totwithin)]]
  dat3_cl = dat3_cl_list[[which.min(dat3_cl_totwithin)]]
  
  dat2_on_dat1_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  dat3_on_dat1_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  
  for (k in 1:K){
    dat2_on_dat1_cent[k,] = apply(dat1[dat2_cl$cluster==k, ],2,mean)
    dat3_on_dat1_cent[k,] = apply(dat1[dat3_cl$cluster==k, ],2,mean)
  }
  
  dat2_on_dat1_dist = apply((dat1 -dat2_on_dat1_cent[dat2_cl$cluster,])^2, 1, sum)
  dat3_on_dat1_dist = apply((dat1 -dat3_on_dat1_cent[dat3_cl$cluster,])^2, 1, sum)
  
  diff_vec = dat2_on_dat1_dist-dat3_on_dat1_dist
  
  sign.test.result = difference.sign.test(diff_vec, alternative="two.sided")
  wilcox.test.result1 = wilcox.test(diff_vec, mu=0, alternative="two.sided")
  wilcox.test.result2 = wilcox.test(x=dat2_on_dat1_dist, y=dat3_on_dat1_dist, mu=0, alternative="two.sided", paired=F)
  
  return(list(sign.test.p = sign.test.result$p.value,
              wilcox.test.p1 = wilcox.test.result1$p.value,
              wilcox.test.p2 = wilcox.test.result2$p.value))
}


library(randtests)

set.seed(7)


save_dir <- "D:/ebd_matrix"

# EBD
or_emb <- readRDS(file.path(save_dir, "or_emb.rds"))

gen1_emb_0_1 <- readRDS(file.path(save_dir, "gen1_emb_0_1.rds"))
gen2_emb_0_1 <- readRDS(file.path(save_dir, "gen2_emb_0_1.rds"))
gen1_1_emb_0_1 <- readRDS(file.path(save_dir, "gen1_1_emb_0_1.rds"))

gen1_emb_0_7 <- readRDS(file.path(save_dir, "gen1_emb_0_7.rds"))
gen2_emb_0_7 <- readRDS(file.path(save_dir, "gen2_emb_0_7.rds"))
gen1_1_emb_0_7 <- readRDS(file.path(save_dir, "gen1_1_emb_0_7.rds"))

gen1_emb_1_5 <- readRDS(file.path(save_dir, "gen1_emb_1_5.rds"))
gen2_emb_1_5 <- readRDS(file.path(save_dir, "gen2_emb_1_5.rds"))
gen1_1_emb_1_5 <- readRDS(file.path(save_dir, "gen1_1_emb_1_5.rds"))

# SVD
SVD_or <- readRDS(file.path(save_dir, "SVD_or.rds"))

SVD_gen1_0_1 <- readRDS(file.path(save_dir, "SVD_gen1_0_1.rds"))
SVD_gen2_0_1 <- readRDS(file.path(save_dir, "SVD_gen2_0_1.rds"))
SVD_gen1_1_0_1 <- readRDS(file.path(save_dir, "SVD_gen1_1_0_1.rds"))

SVD_gen1_0_7 <- readRDS(file.path(save_dir, "SVD_gen1_0_7.rds"))
SVD_gen2_0_7 <- readRDS(file.path(save_dir, "SVD_gen2_0_7.rds"))
SVD_gen1_1_0_7 <- readRDS(file.path(save_dir, "SVD_gen1_1_0_7.rds"))

SVD_gen1_1_5 <- readRDS(file.path(save_dir, "SVD_gen1_1_5.rds"))
SVD_gen2_1_5 <- readRDS(file.path(save_dir, "SVD_gen2_1_5.rds"))
SVD_gen1_1_1_5 <- readRDS(file.path(save_dir, "SVD_gen1_1_1_5.rds"))

dim = 5
alpha = 0.05
k = 3
clrepN = 30

# Reduction
red_or = or_emb%*%SVD_or$v[,1:dim]

red_gen1_0_1 = gen1_emb_0_1%*%SVD_gen1_0_1$v[,1:dim]
red_gen2_0_1 = gen2_emb_0_1%*%SVD_gen2_0_1$v[,1:dim]
red_gen1_1_0_1 = gen1_1_emb_0_1%*%SVD_gen1_1_0_1$v[,1:dim]

red_gen1_0_7 = gen1_emb_0_7%*%SVD_gen1_0_7$v[,1:dim]
red_gen2_0_7 = gen2_emb_0_7%*%SVD_gen2_0_7$v[,1:dim]
red_gen1_1_0_7 = gen1_1_emb_0_7%*%SVD_gen1_1_0_7$v[,1:dim]

red_gen1_1_5 = gen1_emb_1_5%*%SVD_gen1_1_5$v[,1:dim]
red_gen2_1_5 = gen2_emb_1_5%*%SVD_gen2_1_5$v[,1:dim]
red_gen1_1_1_5 = gen1_1_emb_1_5%*%SVD_gen1_1_1_5$v[,1:dim]

################################
# 같은 Temp에서 보기 ###########
################################

### Temp 01
# ori, gen1, gen2
org1g2 = new_try_2(red_or, red_gen1_0_1, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen2
g1org2 = new_try_2(red_gen1_0_1, red_or, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen2
g2g1or = new_try_2(red_gen2_0_1, red_gen1_0_1, red_or, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen1-gen2

# ori, gen1, gen1-1
org1g11 = new_try_2(red_or, red_gen1_0_1, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen1-1
g1org11 = new_try_2(red_gen1_0_1, red_or, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen1-1
g11org1 = new_try_2(red_gen1_1_0_1, red_or, red_gen1_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen1-gen1-1

# ori, gen2, gen1-1
org2g11 = new_try_2(red_or, red_gen2_0_1, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs ori-gen1-1
g11org2 = new_try_2(red_gen1_1_0_1, red_or, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen2-gen1-1
g2org11 = new_try_2(red_gen2_0_1, red_or, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen2-gen1-1

# gen1, gen2, gen1-1
g1g2g11 = new_try_2(red_gen1_0_1, red_gen2_0_1, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen1-gen1-1
g2g1g11 = new_try_2(red_gen2_0_1, red_gen1_0_1, red_gen1_1_0_1, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen2-gen1-1
g11g1g2 = new_try_2(red_gen1_1_0_1, red_gen1_0_1, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen1-1 vs gen2-gen1-1

org1g2$wilcox.test.p2 # 0.755
g1org2$wilcox.test.p2 # 0
g2g1or$wilcox.test.p2 # 0
org1g11$wilcox.test.p2 # 1.09e-165
g1org11$wilcox.test.p2 # 9.96e-26
g11org1$wilcox.test.p2 # 0
g11org2$wilcox.test.p2 # 0
g2org11$wilcox.test.p2 # 5.91e-40
org2g11$wilcox.test.p2 # 6.78e-162
g1g2g11$wilcox.test.p2 # 0
g2g1g11$wilcox.test.p2 # 0
g11g1g2$wilcox.test.p2 # 8.64e-07

### Temp 07
# ori, gen1, gen2
org1g2 = new_try_2(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen2
g1org2 = new_try_2(red_gen1_0_7, red_or, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen2
g2g1or = new_try_2(red_gen2_0_7, red_gen1_0_7, red_or, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen1-gen2

# ori, gen1, gen1-1
org1g11 = new_try_2(red_or, red_gen1_0_7, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen1-1
g1org11 = new_try_2(red_gen1_0_7, red_or, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen1-1
g11org1 = new_try_2(red_gen1_1_0_7, red_or, red_gen1_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen1-gen1-1

# ori, gen2, gen1-1
org2g11 = new_try_2(red_or, red_gen2_0_7, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs ori-gen1-1
g11org2 = new_try_2(red_gen1_1_0_7, red_or, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen2-gen1-1
g2org11 = new_try_2(red_gen2_0_7, red_or, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen2-gen1-1

# gen1, gen2, gen1-1
g1g2g11 = new_try_2(red_gen1_0_7, red_gen2_0_7, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen1-gen1-1
g2g1g11 = new_try_2(red_gen2_0_7, red_gen1_0_7, red_gen1_1_0_7, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen2-gen1-1
g11g1g2 = new_try_2(red_gen1_1_0_7, red_gen1_0_7, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen1-1 vs gen2-gen1-1


org1g2$wilcox.test.p2 # 0.943
g1org2$wilcox.test.p2 # 2.44e-201
g2g1or$wilcox.test.p2 # 9.04e-183
org1g11$wilcox.test.p2 # 2.01e-198
g1org11$wilcox.test.p2 # 3.04e-24
g11org1$wilcox.test.p2 # 0
g11org2$wilcox.test.p2 # 0
g2org11$wilcox.test.p2 # 7.82e-72
org2g11$wilcox.test.p2 # 1.41e-199
g1g2g11$wilcox.test.p2 # 0
g2g1g11$wilcox.test.p2 # 0
g11g1g2$wilcox.test.p2 # 1.39e-25

### Temp 15
# ori, gen1, gen2
org1g2_1_5 = new_try_2(red_or, red_gen1_1_5, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen2
g1org2_1_5 = new_try_2(red_gen1_1_5, red_or, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen2
g2g1or_1_5 = new_try_2(red_gen2_1_5, red_gen1_1_5, red_or, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen1-gen2

# ori, gen1, gen1-1
org1g11_1_5 = new_try_2(red_or, red_gen1_1_5, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs ori-gen1-1
g1org11_1_5 = new_try_2(red_gen1_1_5, red_or, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1 vs gen1-gen1-1
g11org1_1_5 = new_try_2(red_gen1_1_1_5, red_or, red_gen1_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen1-gen1-1

# ori, gen2, gen1-1
org2g11_1_5 = new_try_2(red_or, red_gen2_1_5, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs ori-gen1-1
g11org2_1_5 = new_try_2(red_gen1_1_1_5, red_or, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen1-1 vs gen2-gen1-1
g2org11_1_5 = new_try_2(red_gen2_1_5, red_or, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # ori-gen2 vs gen2-gen1-1

# gen1, gen2, gen1-1
g1g2g11_1_5 = new_try_2(red_gen1_1_5, red_gen2_1_5, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen1-gen1-1
g2g1g11_1_5 = new_try_2(red_gen2_1_5, red_gen1_1_5, red_gen1_1_1_5, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen2 vs gen2-gen1-1
g11g1g2_1_5 = new_try_2(red_gen1_1_1_5, red_gen1_1_5, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T) # gen1-gen1-1 vs gen2-gen1-1

org1g2_1_5$wilcox.test.p2 # 채택
g1org2_1_5$wilcox.test.p2
g2g1or_1_5$wilcox.test.p2
org1g11_1_5$wilcox.test.p2
g1org11_1_5$wilcox.test.p2
g11org1_1_5$wilcox.test.p2
g11org2_1_5$wilcox.test.p2
g2org11_1_5$wilcox.test.p2
org2g11_1_5$wilcox.test.p2
g1g2g11_1_5$wilcox.test.p2
g2g1g11_1_5$wilcox.test.p2
g11g1g2_1_5$wilcox.test.p2


################################
# 다른 Temp에서 보기 ###########
################################

# ori-gen1 vs ori-gen2, Temp 차이두고 보기

# 1. Temp 0.1 vs Temp 0.7
or_gen1_0_1_vs_or_gen2_0_7 = new_try_2(red_or, red_gen1_0_1, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T)

# 2. Temp 0.1 vs Temp 1.5
or_gen1_0_1_vs_or_gen2_1_5 = new_try_2(red_or, red_gen1_0_1, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T)

# 3. Temp 0.7 vs Temp 0.1
or_gen1_0_7_vs_or_gen2_0_1 = new_try_2(red_or, red_gen1_0_7, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T)

# 4. Temp 0.7 vs Temp 1.5
or_gen1_0_7_vs_or_gen2_1_5 = new_try_2(red_or, red_gen1_0_7, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T)

# 5. Temp 1.5 vs Temp 0.1
or_gen1_1_5_vs_or_gen2_0_1 = new_try_2(red_or, red_gen1_1_5, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T)

# 6. Temp 1.5 vs Temp 0.7
or_gen1_1_5_vs_or_gen2_0_7 = new_try_2(red_or, red_gen1_1_5, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T)


or_gen1_0_1_vs_or_gen2_0_7$wilcox.test.p2 # 채택
or_gen1_0_1_vs_or_gen2_1_5$wilcox.test.p2 # 기각
or_gen1_0_7_vs_or_gen2_0_1$wilcox.test.p2 # 기각
or_gen1_0_7_vs_or_gen2_1_5$wilcox.test.p2 # 채택
or_gen1_1_5_vs_or_gen2_0_1$wilcox.test.p2 # 채택
or_gen1_1_5_vs_or_gen2_0_7$wilcox.test.p2 # 채택

################################
# 가설3 차원 다르게 생성은 갖게#
################################

dim1= 3
dim2 = 4
dim3 = 5

# Reduction
red_or = or_emb%*%SVD_or$v[,1:dim1]

red_gen1_0_1 = gen1_emb_0_1%*%SVD_gen1_0_1$v[,1:dim2]
red_gen2_0_1 = gen2_emb_0_1%*%SVD_gen2_0_1$v[,1:dim2]
red_gen1_1_0_1 = gen1_1_emb_0_1%*%SVD_gen1_1_0_1$v[,1:dim2]

red_gen1_0_7 = gen1_emb_0_7%*%SVD_gen1_0_7$v[,1:dim2]
red_gen2_0_7 = gen2_emb_0_7%*%SVD_gen2_0_7$v[,1:dim2]
red_gen1_1_0_7 = gen1_1_emb_0_7%*%SVD_gen1_1_0_7$v[,1:dim2]

red_gen1_1_5 = gen1_emb_1_5%*%SVD_gen1_1_5$v[,1:dim2]
red_gen2_1_5 = gen2_emb_1_5%*%SVD_gen2_1_5$v[,1:dim2]
red_gen1_1_1_5 = gen1_1_emb_1_5%*%SVD_gen1_1_1_5$v[,1:dim2]

### ori-gen1 vs ori-gen2

org1g2_ = new_try_2(red_or, red_gen1_0_1, red_gen2_0_1, alpha, K=k, repN=clrepN, standardize=T) # temp 0.1
org1g2__ = new_try_2(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=k, repN=clrepN, standardize=T) # temp 0.7
org1g2___ = new_try_2(red_or, red_gen1_1_5, red_gen2_1_5, alpha, K=k, repN=clrepN, standardize=T) # temp 1.5


org1g2_$wilcox.test.p2 # 0.978 , 0.924
org1g2__$wilcox.test.p2 # 0.909, 0.952
org1g2___$wilcox.test.p2 # 0.384, 0.712


