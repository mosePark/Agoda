rm(list = ls(all=TRUE))
library(mvtnorm)
library(randtests)

### test function is modified.

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
  
  sign.test.result = difference.sign.test(diff_vec, alternative="left.sided")
  wilcox.test.result1 = wilcox.test(diff_vec, mu=0, alternative="less")
  wilcox.test.result2 = wilcox.test(x=dat2_on_dat1_dist, y=dat3_on_dat1_dist, mu=0, alternative="less", paired=F)
  #permTS.test.result = permTS(dat2_on_dat1_dist, dat3_on_dat1_dist, exact=T)
  #permTS.test.result = 1
  
  return(list(sign.test.p = sign.test.result$p.value, wilcox.test.p1 = wilcox.test.result1$p.value, wilcox.test.p2 = wilcox.test.result2$p.value ))
}


## example: alternative hyp. scenario
cent1 = c(sqrt(3), 1); cent2 = c(-sqrt(3),1); cent3 = c(0,-2)
# 1st; 2nd, and 3rd sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2)/2,0,0,sqrt(2)), nrow=2)


realization = 1000


set.seed(7)
###################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 포커스 #########################################
###################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  Y2 = 1.0*Z2 + .2*matrix(rnorm(n*2), ncol=2)
  Y3 = 1.0*Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  K1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  K2 = 1.0*Z2 + .2*matrix(rnorm(n*2), ncol=2)
  K3 = 1.0*Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  #result
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}
# layout(matrix(c(1,2,3), nrow=1))


hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 포커스 ##########################################
####################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = 3*rmvnorm(n, mean= cent1, sigma = sig1)
  Y2 = 3*rmvnorm(n, mean= cent2, sigma = sig2)
  Y3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  K1 = 3*rmvnorm(n, mean= cent1, sigma = sig1)
  K2 = 3*rmvnorm(n, mean= cent2, sigma = sig2)
  K3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

####################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 평행이동 ########################################
####################################################

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = Z1 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  Y2 = Z2 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  Y3 = Z3 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  
  K1 = Z1 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  K2 = Z2 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  K3 = Z3 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 평행이동 ########################################
###################################################

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = rmvnorm(n, mean= cent1, sigma = sig1) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  Y2 = rmvnorm(n, mean= cent2, sigma = sig2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  Y3 = rmvnorm(n, mean= cent3, sigma = sig3) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  
  K1 = rmvnorm(n, mean= cent1, sigma = sig1) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  K2 = rmvnorm(n, mean= cent2, sigma = sig2) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  K3 = rmvnorm(n, mean= cent3, sigma = sig3) + matrix(c(-3, 2), nrow = n, ncol = 2, byrow = TRUE)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 동일 회전 #######################################
####################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

rotmat = matrix(c(cos(pi/3), -sin(pi/3), sin(pi/3), cos(pi/3)), nrow=2, byrow=T)

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = Z1%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  Y2 = Z2%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  Y3 = Z3%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  
  K1 = Z1%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  K2 = Z2%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  K3 = Z3%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  #result
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}
# layout(matrix(c(1,2,3), nrow=1))


hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 동일 회전 #######################################
####################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
  Y2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
  Y3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat
  
  K1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
  K2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
  K3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

####################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 다른 회전 #######################################
####################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

rotmat = matrix(c(cos(pi/3), -sin(pi/3), sin(pi/3), cos(pi/3)), nrow=2, byrow=T)
rotmat2 = matrix(c(cos(-2*pi/3), -sin(-2*pi/3), sin(-2*pi/3), cos(-2*pi/3)), nrow=2, byrow=T)

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = Z1%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  Y2 = Z2%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  Y3 = Z3%*%rotmat + .2*matrix(rnorm(n*2), ncol=2)
  
  K1 = Z1%*%rotmat2 + .2*matrix(rnorm(n*2), ncol=2)
  K2 = Z2%*%rotmat2 + .2*matrix(rnorm(n*2), ncol=2)
  K3 = Z3%*%rotmat2 + .2*matrix(rnorm(n*2), ncol=2)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  #result
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}
# layout(matrix(c(1,2,3), nrow=1))


hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 다른 회전 #######################################
####################################################
n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
  Y2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
  Y3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat
  
  K1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat2
  K2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat2
  K3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat2
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

####################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 도넛 변형 #######################################
####################################################
gen_donut <- function(n, inner_radius = 0.5, outer_radius = 1.5, centroid = c(0, 0), noise_level = 0.2) {
  theta <- runif(n, 0, 2*pi)
  r <- sqrt(runif(n, inner_radius^2, outer_radius^2))
  x <- r * cos(theta)
  y <- r * sin(theta)
  
  donut_data <- cbind(x, y) + matrix(rep(centroid, n), ncol=2, byrow=TRUE)
  
  # noise <- noise_level * matrix(rnorm(n*2), ncol=2)
  # donut_data <- donut_data + noise
  
  return(donut_data)
}

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent1) + .2*matrix(rnorm(n*2), ncol=2)
  Y2 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent2) + .2*matrix(rnorm(n*2), ncol=2)
  Y3 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent3) + .2*matrix(rnorm(n*2), ncol=2)
  
  K1 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent1) + .2*matrix(rnorm(n*2), ncol=2)
  K2 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent2) + .2*matrix(rnorm(n*2), ncol=2)
  K3 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent3) + .2*matrix(rnorm(n*2), ncol=2)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  #result
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 도넛 변형 #######################################
####################################################

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent1)
  Y2 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent2)
  Y3 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent3)
  
  K1 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent1)
  K2 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent2)
  K3 = gen_donut(n, inner_radius = 0.5, outer_radius = 1.5, centroid = cent3)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

####################################################
## example: null hyp. scenario under "PAIRED MODEL"
## 나선 변형 #######################################
####################################################
gen_spiral <- function(n, turns = 3, spread = 0.5, centroid = c(0, 0), noise_level = 0.2) {
  
  theta <- seq(0, 2*pi*turns, length.out = n)
  r <- spread * theta
  
  x <- r * cos(theta)
  y <- r * sin(theta)
  
  spiral_data <- cbind(x, y) + matrix(rep(centroid, n), ncol=2, byrow=TRUE)
  
  # noise <- noise_level * matrix(rnorm(n*2), ncol=2)
  # spiral_data <- spiral_data + noise
  
  return(spiral_data)
}

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2*matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2*matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2*matrix(rnorm(n*2), ncol=2)
  
  Y1 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent1) + .2*matrix(rnorm(n*2), ncol=2)
  Y2 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent2) + .2*matrix(rnorm(n*2), ncol=2)
  Y3 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent3) + .2*matrix(rnorm(n*2), ncol=2)
  
  K1 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent1) + .2*matrix(rnorm(n*2), ncol=2)
  K2 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent2) + .2*matrix(rnorm(n*2), ncol=2)
  K3 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent3) + .2*matrix(rnorm(n*2), ncol=2)
  
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  #result
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)

set.seed(7)
####################################################
## example: null hyp. scenario under "UNPAIRED MODEL"
## 나선 변형 #######################################
####################################################

n=1000

sign.pvec = NULL
wilcox.pvec1 = NULL
wilcox.pvec2 = NULL

for (i in 1:realization){
  set.seed(i)
  
  #Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  #Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  #Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = rmvnorm(n, mean= cent1, sigma = sig1)
  X2 = rmvnorm(n, mean= cent2, sigma = sig2)
  X3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  Y1 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent1)
  Y2 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent2)
  Y3 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent3)
  
  K1 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent1)
  K2 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent2)
  K3 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent3)
  
  raw_dat1 = rbind(X1, X2, X3)
  raw_dat2 = rbind(Y1, Y2, Y3)
  raw_dat3 = rbind(K1, K2, K3)
  
  result = new_try_2(raw_dat1, raw_dat2, raw_dat3, K=2, alpha = .05, standardize=T)
  
  sign.pvec[i] = result$sign.test.p
  wilcox.pvec1[i] = result$wilcox.test.p1
  wilcox.pvec2[i] = result$wilcox.test.p2
  
  cat(i)
}

hist(sign.pvec, breaks=20)
hist(wilcox.pvec1, breaks=20)
hist(wilcox.pvec2, breaks=20)
