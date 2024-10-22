gen_spiral <- function(n, turns = 3, spread = 0.5, centroid = c(0, 0), noise_level = 0.2) {
  
  theta <- seq(0, 2*pi*turns, length.out = n)
  
  x <- r * cos(theta)
  y <- r * sin(theta)
  
  spiral_data <- cbind(x, y) + matrix(rep(centroid, n), ncol=2, byrow=TRUE)
  
  noise <- noise_level * matrix(rnorm(n*2), ncol=2)
  spiral_data <- spiral_data + noise
  
  return(spiral_data)
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

spiral_X1 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent1, noise_level = 0.2)
spiral_X2 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent2, noise_level = 0.2)
spiral_X3 = gen_spiral(n, turns = 3, spread = 0.2, centroid = cent3, noise_level = 0.2)

dat2 = rbind(spiral_X1, spiral_X2, spiral_X3)
