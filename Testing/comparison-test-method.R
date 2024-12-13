eqtest = function(raw_dat1, raw_dat2, raw_dat3, alpha, K=3, repN=10, standardize=T, sample_size = 10000){
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
  
  sample_d2 <- sample(dat2_on_dat1_dist, sample_size)
  sample_d3 <- sample(dat3_on_dat1_dist, sample_size)
  
  sample <- rbind(matrix(sample_d2, ncol = 1), matrix(sample_d3, ncol = 1))
  
  d_mat <- dist(sample)
  
  # eqdist.etest
  result <- eqdist.etest(d_mat, sizes = c(sample_size, sample_size), distance = TRUE, R = 1000)
  
  return(list(call = result$call,
              method = result$method,
              stat = result$statistic,
              p = result$p.value
  )
  )
}

bdtest = function(raw_dat1, raw_dat2, raw_dat3, alpha, K=3, repN=10, standardize=T, sample_size = 3000){
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
  
  sample_d2 <- sample(dat2_on_dat1_dist, sample_size)
  sample_d3 <- sample(dat3_on_dat1_dist, sample_size)
  
  sample <- rbind(matrix(sample_d2, ncol = 1), matrix(sample_d3, ncol = 1))
  
  d_mat <- dist(sample)
  
  # eqdist.etest
  result <- bd.test(sample_d2, sample_d3)

  return(list(
              stat = result$statistic,
              p = result$p.value
  )
  )
}


library(randtests)
library(energy)
library(Ball)


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

dim = 2
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
# 같은 Temp 0.7에서 보기 #######
################################

### Temp 07
# ori, gen1, gen2
'
eqtest(raw_dat1, raw_dat2, raw_dat3, alpha, K=3, repN=10, standardize=T, sample_size = 10000)
'

org1g2 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=2, repN=clrepN, standardize=T, sample_size = 5000)
org1g23 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000)
org1g24 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=4, repN=clrepN, standardize=T, sample_size = 5000)
org1g25 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=5, repN=clrepN, standardize=T, sample_size = 5000)
org1g26 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=6, repN=clrepN, standardize=T, sample_size = 5000)
org1g27 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=7, repN=clrepN, standardize=T, sample_size = 5000)
org1g28 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=8, repN=clrepN, standardize=T, sample_size = 5000)
org1g29 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=9, repN=clrepN, standardize=T, sample_size = 5000)

org1g2$p
org1g23$p
org1g24$p
org1g25$p
org1g26$p
org1g27$p
org1g28$p
org1g29$p

' sample 3000
> org1g2$p
[1] 1

> org1g23$p
[1] 1

> org1g24$p
[1] 0.3636364

> org1g25$p
[1] 0.09090909

> org1g26$p
[1] 0.6363636

> org1g27$p
[1] 0.1818182

> org1g28$p
[1] 0.9090909

> org1g29$p
[1] 0.7272727
'

' sample 7000

> org1g2$p
[1] 0.7272727

> org1g23$p
[1] 0.2727273

> org1g24$p
[1] 0.9090909

> org1g25$p
[1] 0.3636364

> org1g26$p
[1] 0.8181818

> org1g27$p
[1] 0.7272727

> org1g28$p
[1] 0.9090909

> org1g29$p
[1] 0.2727273

뭔가 유의확률 값이 이상한데 R=100 (부트스트랩 반복수를 늘려보자.)
'

org1g2 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=2, repN=clrepN, standardize=T, sample_size = 5000)
org1g23 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000)
org1g24 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=4, repN=clrepN, standardize=T, sample_size = 5000)
org1g25 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=5, repN=clrepN, standardize=T, sample_size = 5000)
org1g26 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=6, repN=clrepN, standardize=T, sample_size = 5000)
org1g27 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=7, repN=clrepN, standardize=T, sample_size = 5000)
org1g28 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=8, repN=clrepN, standardize=T, sample_size = 5000)
org1g29 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=9, repN=clrepN, standardize=T, sample_size = 5000)

org1g2$p
org1g23$p
org1g24$p
org1g25$p
org1g26$p
org1g27$p
org1g28$p
org1g29$p

' sample 5000 기준
> org1g2$p
[1] 0.04950495

> org1g23$p
[1] 0.6039604

> org1g24$p
[1] 0.7425743

> org1g25$p
[1] 0.06930693

> org1g26$p
[1] 0.1683168

> org1g27$p
[1] 0.3465347

> org1g28$p
[1] 0.4455446

> org1g29$p
[1] 0.1881188

부트스트랩 1000번일 경우도 해보자.
'

org1g2 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=2, repN=clrepN, standardize=T, sample_size = 5000)
org1g23 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000)
org1g24 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=4, repN=clrepN, standardize=T, sample_size = 5000)
org1g25 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=5, repN=clrepN, standardize=T, sample_size = 5000)
org1g26 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=6, repN=clrepN, standardize=T, sample_size = 5000)
org1g27 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=7, repN=clrepN, standardize=T, sample_size = 5000)
org1g28 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=8, repN=clrepN, standardize=T, sample_size = 5000)
org1g29 = eqtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=9, repN=clrepN, standardize=T, sample_size = 5000)

org1g2$p
org1g23$p
org1g24$p
org1g25$p
org1g26$p
org1g27$p
org1g28$p
org1g29$p

'
> org1g2$p
[1] 0.993007

> org1g23$p
[1] 0.3146853

> org1g24$p
[1] 0.04495504

> org1g25$p
[1] 0.1258741

> org1g26$p
[1] 0.3456543

> org1g27$p
[1] 0.6283716

> org1g28$p
[1] 0.7172827

> org1g29$p
[1] 0.5524476
'



### Temp 07
# 다른것들도 보기

g1org2 =  eqtest(red_gen1_0_7, red_or, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs gen1-gen2
g2g1or = eqtest(red_gen2_0_7, red_gen1_0_7, red_or, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000)  # ori-gen2 vs gen1-gen2

# ori, gen1, gen1-1

org1g11 = eqtest(red_or, red_gen1_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs ori-gen1-1
g1org11 = eqtest(red_gen1_0_7, red_or, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs gen1-gen1-1
g11org1 = eqtest(red_gen1_1_0_7, red_or, red_gen1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1-1 vs gen1-gen1-1

# ori, gen2, gen1-1

org2g11 = eqtest(red_or, red_gen2_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen2 vs ori-gen1-1
g11org2 = eqtest(red_gen1_1_0_7, red_or, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1-1 vs gen2-gen1-1
g2org11 = eqtest(red_gen2_0_7, red_or, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen2 vs gen2-gen1-1


# gen1, gen2, gen1-1

g1g2g11 = eqtest(red_gen1_0_7, red_gen2_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen2 vs gen1-gen1-1
g2g1g11 = eqtest(red_gen2_0_7, red_gen1_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen2 vs gen2-gen1-1
g11g1g2 = eqtest(red_gen1_1_0_7, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen1-1 vs gen2-gen1-1

org1g2$p
g1org2$p
g2g1or$p
org1g11$p
g1org11$p
g11org1$p
g11org2$p
g2org11$p
org2g11$p
g1g2g11$p
g2g1g11$p
g11g1g2$p

'
> org1g2$p
[1] 0.993007

> g1org2$p
[1] 0.000999001

> g2g1or$p
[1] 0.000999001

> org1g11$p
[1] 0.06793207

> g1org11$p
[1] 0.000999001

> g11org1$p
[1] 0.000999001

> g11org2$p
[1] 0.000999001

> g2org11$p
[1] 0.000999001

> org2g11$p
[1] 0.02997003

> g1g2g11$p
[1] 0.5244755

> g2g1g11$p
[1] 0.000999001

> g11g1g2$p
[1] 0.000999001

'

### Temp 07
# Ball test

'
bdtest(raw_dat1, raw_dat2, raw_dat3, alpha, K=3, repN=10, standardize=T, sample_size = 3000)
'
# ori, gen1, gen2

org1g2 = bdtest(red_or, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs ori-gen2
g1org2 =  bdtest(red_gen1_0_7, red_or, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs gen1-gen2
g2g1or = bdtest(red_gen2_0_7, red_gen1_0_7, red_or, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000)  # ori-gen2 vs gen1-gen2

# ori, gen1, gen1-1

org1g11 = bdtest(red_or, red_gen1_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs ori-gen1-1
g1org11 = bdtest(red_gen1_0_7, red_or, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1 vs gen1-gen1-1
g11org1 = bdtest(red_gen1_1_0_7, red_or, red_gen1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1-1 vs gen1-gen1-1

# ori, gen2, gen1-1

org2g11 = bdtest(red_or, red_gen2_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen2 vs ori-gen1-1
g11org2 = bdtest(red_gen1_1_0_7, red_or, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen1-1 vs gen2-gen1-1
g2org11 = bdtest(red_gen2_0_7, red_or, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # ori-gen2 vs gen2-gen1-1

# gen1, gen2, gen1-1
g1g2g11 = bdtest(red_gen1_0_7, red_gen2_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen2 vs gen1-gen1-1
g2g1g11 = bdtest(red_gen2_0_7, red_gen1_0_7, red_gen1_1_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen2 vs gen2-gen1-1
g11g1g2 = bdtest(red_gen1_1_0_7, red_gen1_0_7, red_gen2_0_7, alpha, K=3, repN=clrepN, standardize=T, sample_size = 5000) # gen1-gen1-1 vs gen2-gen1-1

cat("org1g2 p-value:", org1g2$p, "\n")
cat("g1org2 p-value:", g1org2$p, "\n")
cat("g2g1or p-value:", g2g1or$p, "\n")

cat("org1g11 p-value:", org1g11$p, "\n")
cat("g1org11 p-value:", g1org11$p, "\n")
cat("g11org1 p-value:", g11org1$p, "\n")

cat("org2g11 p-value:", org2g11$p, "\n")
cat("g11org2 p-value:", g11org2$p, "\n")
cat("g2org11 p-value:", g2org11$p, "\n")

cat("g1g2g11 p-value:", g1g2g11$p, "\n")
cat("g2g1g11 p-value:", g2g1g11$p, "\n")
cat("g11g1g2 p-value:", g11g1g2$p, "\n")



