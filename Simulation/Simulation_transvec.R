library(mvtnorm)

# 1st center; 2nd center; 3rd center
cent1 = c(sqrt(3), 1); cent2 = c(-sqrt(3),1); cent3 = c(0,-2)
# 1st; 2nd, and 3rd sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2)/2,0,0,sqrt(2)), nrow=2)

n=1000

realization = 1000

for (i in 1:realization) {
  Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
  Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
  Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
  
  X1 = Z1 + .2 * matrix(rnorm(n*2), ncol=2)
  X2 = Z2 + .2 * matrix(rnorm(n*2), ncol=2)
  X3 = Z3 + .2 * matrix(rnorm(n*2), ncol=2)
  
  # dat2 vector translation
  Y1 = Z1 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-5, 8), nrow = n, ncol = 2, byrow = TRUE)
  Y2 = Z2 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-5, 8), nrow = n, ncol = 2, byrow = TRUE)
  Y3 = Z3 + .2 * matrix(rnorm(n*2), ncol=2) + matrix(c(-5, 8), nrow = n, ncol = 2, byrow = TRUE)
  
  dat1 = rbind(X1, X2, X3)
  dat2 = rbind(Y1, Y2, Y3)
  
  result = new_try(dat1, dat2, K=3, alpha = .05, itrN=1000)
  
  pval = result$p
  
  pvec[i] = pval
  cat(i)
}
