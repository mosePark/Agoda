library(mvtnorm)
library(colorspace)

## color

ylorrd_colors <- hcl.colors(5, "YlOrRd")
ylgnbu_colors <- hcl.colors(5, "YlGnBu")

# ex
barplot(1:3, col = ylorrd_colors, main = "YlOrRd First 3 Colors")
barplot(1:3, col = ylgnbu_colors, main = "YlGnBu First 3 Colors")

print(ylorrd_colors[1:3])
print(ylgnbu_colors[1:3])


## Centroid
cent1 = c(sqrt(3), 1); cent2 = c(-sqrt(3),1); cent3 = c(0,-2)

# Sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2)/2,0,0,sqrt(2)), nrow=2)


################################################################################
### Null Scenario ##############################################################
################################################################################


n = 100
rotmat = matrix(c(cos(pi/3), -sin(pi/3), sin(pi/3), cos(pi/3)), nrow=2, byrow=T)

X1 = rmvnorm(n, mean = cent1, sigma = sig1)
X2 = rmvnorm(n, mean = cent2, sigma = sig2)
X3 = rmvnorm(n, mean = cent3, sigma = sig3)

Y1 = rmvnorm(n, mean = cent1, sigma = sig1) %*% rotmat
Y2 = rmvnorm(n, mean = cent2, sigma = sig2) %*% rotmat
Y3 = rmvnorm(n, mean = cent3, sigma = sig3) %*% rotmat

K1 = rmvnorm(n, mean = cent1, sigma = sig1) %*% rotmat
K2 = rmvnorm(n, mean = cent2, sigma = sig2) %*% rotmat
K3 = rmvnorm(n, mean = cent3, sigma = sig3) %*% rotmat

raw_dat1 = rbind(X1, X2, X3)

# Perturb for better visualization
set.seed(1)
purt.size = n * 0.2
purt.ind = sample(1:(3 * n), size = 3 * purt.size)
raw_dat1[purt.ind[1:purt.size], ] = 3 * rmvnorm(purt.size, mean = cent1, sigma = sig1)
raw_dat1[purt.ind[1:purt.size + purt.size], ] = 2 * rmvnorm(purt.size, mean = cent2, sigma = sig2)
raw_dat1[purt.ind[1:purt.size + purt.size * 2], ] = rmvnorm(purt.size, mean = cent3, sigma = sig3)

raw_dat2 = rbind(Y1, Y2, Y3)
raw_dat3 = rbind(K1, K2, K3)

# Create perturbed indices for raw_dat3
purt.size = n * 0.2
indset = 1:(3 * n)
purt.ind2 = sample(1:(3 * n), size = 3 * purt.size)

ind1 = union(setdiff(1:n, purt.ind2), purt.ind2[1:purt.size])
ind2 = union(setdiff((n + 1):(2 * n), purt.ind2), purt.ind2[1:purt.size + purt.size])
ind3 = union(setdiff((2 * n + 1):(3 * n), purt.ind2), purt.ind2[1:purt.size + 2 * purt.size])

raw_dat3[ind1, ] = rmvnorm(length(ind1), mean = cent1, sigma = sig1) %*% rotmat
raw_dat3[ind2, ] = rmvnorm(length(ind2), mean = cent2, sigma = sig2) %*% rotmat
raw_dat3[ind3, ] = rmvnorm(length(ind3), mean = cent3, sigma = sig3) %*% rotmat


# Centroid 계산
dat2_centroids <- matrix(NA, nrow = 3, ncol = ncol(raw_dat2))
dat3_centroids <- matrix(NA, nrow = 3, ncol = ncol(raw_dat3))

dat2_centroids[1, ] <- colMeans(raw_dat2[1:n, ])
dat2_centroids[2, ] <- colMeans(raw_dat2[(n + 1):(2 * n), ])
dat2_centroids[3, ] <- colMeans(raw_dat2[(2 * n + 1):(3 * n), ])

dat3_centroids[1, ] <- colMeans(raw_dat3[ind1, ])
dat3_centroids[2, ] <- colMeans(raw_dat3[ind2, ])
dat3_centroids[3, ] <- colMeans(raw_dat3[ind3, ])

# 결과 출력
print("Centroids for raw_dat2:")
print(dat2_centroids)

print("Centroids for raw_dat3:")
print(dat3_centroids)

################################################################################
### Alter Scenario #############################################################
################################################################################

# Create new indices for raw_dat3_new
new_ind = sample(1:(3 * n))
new_ind1 = new_ind[1:n]
new_ind2 = new_ind[(n + 1):(2 * n)]
new_ind3 = new_ind[(2 * n + 1):(3 * n)]

# Generate raw_dat3_new with rotation applied to selected indices
raw_dat3_new = raw_dat3
raw_dat3_new[new_ind1, ] = rmvnorm(n, mean = cent1, sigma = sig1) %*% rotmat
raw_dat3_new[new_ind2, ] = rmvnorm(n, mean = cent2, sigma = sig2) %*% rotmat
raw_dat3_new[new_ind3, ] = rmvnorm(n, mean = cent3, sigma = sig3) %*% rotmat



# Centroid 계산
dat2_centroids <- matrix(NA, nrow = 3, ncol = ncol(raw_dat2))
dat3_centroids <- matrix(NA, nrow = 3, ncol = ncol(raw_dat3))

dat2_centroids[1, ] <- colMeans(raw_dat2[1:n, ])
dat2_centroids[2, ] <- colMeans(raw_dat2[(n + 1):(2 * n), ])
dat2_centroids[3, ] <- colMeans(raw_dat2[(2 * n + 1):(3 * n), ])

dat3_centroids[1, ] <- colMeans(raw_dat3_new[new_ind1, ])
dat3_centroids[2, ] <- colMeans(raw_dat3_new[new_ind2, ])
dat3_centroids[3, ] <- colMeans(raw_dat3_new[new_ind3, ])

# 결과 출력
print("Centroids for raw_dat2:")
print(dat2_centroids)

print("Centroids for raw_dat3:")
print(dat3_centroids)












################################################################################
# NULL 그리기 ##################################################################
################################################################################

# tiff("NULL.tiff", width = 2800, height = 2800, res = 800)

# par(mar = c(2,2,2,2))
# par(mar = c(5, 4, 4, 2) + 0.1)


par(mfrow = c(1, 1))

# 1 by 1
par(fig = c(0, 0.33, 0.5, 1), new = TRUE)
plot(raw_dat2, xlab='X1', ylab='X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat2[1:n, ], cex=1.2, col="#7D0025")
points(raw_dat2[(n + 1):(2 * n), ], cex=1.2, col="#DA3500")
points(raw_dat2[(2 * n + 1):(3 * n),], cex=1.2, col="#F39300")
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 1 by 2
par(fig = c(0.33, 0.66, 0.5, 1), new = TRUE)
plot(raw_dat1, cex=1.2, xlab='X1', ylab='X2', xlim=c(-6, 6), ylim=c(-6, 6))
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 1 by 3
par(fig = c(0.66, 1, 0.5, 1), new = TRUE)
plot(raw_dat3, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat3[ind1,], cex=1.2, col="#26185F")
points(raw_dat3[ind2,], cex=1.2, col="#007EB3")
points(raw_dat3[ind3,], cex=1.2, col="#18BDB0")
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 2 by 1
par(fig = c(0.2, 0.5, 0, 0.5), new = TRUE)
plot(raw_dat1, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat1[1:n, ], cex=1.2, col="#7D0025")
points(raw_dat1[(n + 1):(2 * n), ], cex=1.2, col="#DA3500")
points(raw_dat1[(2 * n + 1):(3 * n), ], cex=1.2, col="#F39300")

points(dat2_centroids, col = "black", pch = 17, cex = 2) # 검정색 삼각형으로 표시

axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 2 by 2
par(fig = c(0.5, 0.8, 0, 0.5), new = TRUE)
plot(raw_dat1, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat1[ind1, ], cex=1.2, col="#26185F")
points(raw_dat1[ind2, ], cex=1.2, col="#007EB3")
points(raw_dat1[ind3, ], cex=1.2, col="#18BDB0")
points(dat3_centroids, col = "black", pch = 17, cex = 2)
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)



dev.off()


################################################################################
# Alter 그리기 #################################################################
################################################################################

par(mfrow = c(1, 1))

# 1 by 1
par(fig = c(0, 0.33, 0.5, 1), new = TRUE)
plot(raw_dat2, xlab='X1', ylab='X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat2[1:n, ], cex=1.2, col="#7D0025")
points(raw_dat2[(n + 1):(2 * n), ], cex=1.2, col="#DA3500")
points(raw_dat2[(2 * n + 1):(3 * n),], cex=1.2, col="#F39300")
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 1 by 2
par(fig = c(0.33, 0.66, 0.5, 1), new = TRUE)
plot(raw_dat1, cex=1.2, xlab='X1', ylab='X2', xlim=c(-6, 6), ylim=c(-6, 6))
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 1 by 3 (new)
par(fig = c(0.66, 1, 0.5, 1), new = TRUE)
plot(raw_dat3_new, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat3_new[new_ind1, ], cex=1.2, col="#26185F")
points(raw_dat3_new[new_ind2, ], cex=1.2, col="#007EB3")
points(raw_dat3_new[new_ind3, ], cex=1.2, col="#18BDB0")
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)

# 2 by 1
par(fig = c(0.2, 0.5, 0, 0.5), new = TRUE)
plot(raw_dat1, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat1[1:n, ], cex=1.2, col="#7D0025")
points(raw_dat1[(n + 1):(2 * n), ], cex=1.2, col="#DA3500")
points(raw_dat1[(2 * n + 1):(3 * n), ], cex=1.2, col="#F39300")

points(dat2_centroids, col = "black", pch = 17, cex = 2) # 검정색 삼각형으로 표시

axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)



# 2 by 2
par(fig = c(0.5, 0.8, 0, 0.5), new = TRUE)
plot(raw_dat1, xlab ='X1', ylab = 'X2', xlim=c(-6, 6), ylim=c(-6, 6))
points(raw_dat1[new_ind1, ], cex=1.2, col="#26185F")
points(raw_dat1[new_ind2, ], cex=1.2, col="#007EB3")
points(raw_dat1[new_ind3, ], cex=1.2, col="#18BDB0")
points(dat3_centroids, col = "black", pch = 17, cex = 2)
axis(1, lwd = 2.5)
axis(2, lwd = 2.5)
box(lwd = 2.5)



dev.off()

