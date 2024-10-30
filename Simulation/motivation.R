## example: alternative hyp. scenario
cent1 = c(sqrt(4), 1); cent2 = c(-sqrt(4),1); cent3 = c(0,-5)
# 1st; 2nd, and 3rd sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2),0,0,sqrt(2)), nrow=2)


###############################################################################
### NULL. Scenario ############################################################
###############################################################################

n = 30
rotmat = matrix(c(cos(pi/3), -sin(pi/3), sin(pi/3), cos(pi/3)), nrow=2, byrow=T)

X1 = 3 * rmvnorm(n, mean = cent1, sigma = sig1)
X2 = 2 * rmvnorm(n, mean = cent2, sigma = sig2)
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

layout(matrix(1:6, nrow=2, byrow=T))

# Plot raw_dat1 (1 by 1)
plot(raw_dat1)

# Plot raw_dat2 (1 by 2)
plot(raw_dat2)
points(raw_dat2[1:n, ], pch=16, cex=1.5, col="red")
points(raw_dat2[(n + 1):(2 * n), ], pch=15, cex=1.5, col="pink")
points(raw_dat2[(2 * n + 1):(3 * n), ], pch=17, cex=1.5, col="darkred")
text(raw_dat2[1:n, ], labels = 1:n, pos = 3, cex = 0.9)
text(raw_dat2[(n + 1):(2 * n), ], labels = (n + 1):(2 * n), pos = 3, cex = 0.9)
text(raw_dat2[(2 * n + 1):(3 * n), ], labels = (2 * n + 1):(3 * n), pos = 3, cex = 0.9)

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

# Plot raw_dat3 (1 by 3)
plot(raw_dat3)
points(raw_dat3[ind1, ], pch=16, cex=1.5, col="green")
points(raw_dat3[ind2, ], pch=15, cex=1.5, col="lightgreen")
points(raw_dat3[ind3, ], pch=17, cex=1.5, col="darkgreen")
text(raw_dat3[ind1, ], labels = ind1, pos = 3, cex = 0.9)
text(raw_dat3[ind2, ], labels = ind2, pos = 3, cex = 0.9)
text(raw_dat3[ind3, ], labels = ind3, pos = 3, cex = 0.9)

# Plot raw_dat1 (2 by 1)
plot(raw_dat1)

# Plot raw_dat1 (2 by 2)
plot(raw_dat1)
points(raw_dat1[1:n, ], pch=16, cex=1.5, col="red")
points(raw_dat1[(n + 1):(2 * n), ], pch=15, cex=1.5, col="pink")
points(raw_dat1[(2 * n + 1):(3 * n), ], pch=17, cex=1.5, col="darkred")
text(raw_dat1[1:n, ], labels = 1:n, pos = 3, cex = 0.9)
text(raw_dat1[(n + 1):(2 * n), ], labels = (n + 1):(2 * n), pos = 3, cex = 0.9)
text(raw_dat1[(2 * n + 1):(3 * n), ], labels = (2 * n + 1):(3 * n), pos = 3, cex = 0.9)

# Plot raw_dat1 (2 by 3) with larger points and indices
plot(raw_dat1)
points(raw_dat1[ind1, ], pch=16, cex=1.5, col="green")
points(raw_dat1[ind2, ], pch=15, cex=1.5, col="lightgreen")
points(raw_dat1[ind3, ], pch=17, cex=1.5, col="darkgreen")
text(raw_dat1[ind1, ], labels = ind1, pos = 3, cex = 0.9)
text(raw_dat1[ind2, ], labels = ind2, pos = 3, cex = 0.9)
text(raw_dat1[ind3, ], labels = ind3, pos = 3, cex = 0.9)

###############################################################################
### Alt. Scenario##############################################################
###############################################################################

layout(matrix(1:6, nrow=2, byrow=T))

# Plot raw_dat1 (1 by 1)
plot(raw_dat1)

# Plot raw_dat2 (1 by 2)
plot(raw_dat2)
points(raw_dat2[1:n, ], pch=16, cex=1.5, col="red")
points(raw_dat2[(n + 1):(2 * n), ], pch=15, cex=1.5, col="pink")
points(raw_dat2[(2 * n + 1):(3 * n), ], pch=17, cex=1.5, col="darkred")
text(raw_dat2[1:n, ], labels = 1:n, pos = 3, cex = 0.8)
text(raw_dat2[(n + 1):(2 * n), ], labels = (n + 1):(2 * n), pos = 3, cex = 0.8)
text(raw_dat2[(2 * n + 1):(3 * n), ], labels = (2 * n + 1):(3 * n), pos = 3, cex = 0.8)

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

# Plot raw_dat3_new (1 by 3)
plot(raw_dat3_new)
points(raw_dat3_new[new_ind1, ], pch=16, cex=1.5, col="green")
points(raw_dat3_new[new_ind2, ], pch=15, cex=1.5, col="lightgreen")
points(raw_dat3_new[new_ind3, ], pch=17, cex=1.5, col="darkgreen")
text(raw_dat3_new[new_ind1, ], labels = new_ind1, pos = 3, cex = 0.8)
text(raw_dat3_new[new_ind2, ], labels = new_ind2, pos = 3, cex = 0.8)
text(raw_dat3_new[new_ind3, ], labels = new_ind3, pos = 3, cex = 0.8)

# Plot raw_dat1 (2 by 1)
plot(raw_dat1)

# Plot raw_dat1 (2 by 2)
plot(raw_dat1)
points(raw_dat1[1:n, ], pch=16, cex=1.5, col="red")
points(raw_dat1[(n + 1):(2 * n), ], pch=15, cex=1.5, col="pink")
points(raw_dat1[(2 * n + 1):(3 * n), ], pch=17, cex=1.5, col="darkred")
text(raw_dat1[1:n, ], labels = 1:n, pos = 3, cex = 0.8)
text(raw_dat1[(n + 1):(2 * n), ], labels = (n + 1):(2 * n), pos = 3, cex = 0.8)
text(raw_dat1[(2 * n + 1):(3 * n), ], labels = (2 * n + 1):(3 * n), pos = 3, cex = 0.8)

# Plot raw_dat1 (2 by 3)
plot(raw_dat1)
points(raw_dat1[new_ind1, ], pch=16, cex=1.5, col="green")
points(raw_dat1[new_ind2, ], pch=15, cex=1.5, col="lightgreen")
points(raw_dat1[new_ind3, ], pch=17, cex=1.5, col="darkgreen")
text(raw_dat1[new_ind1, ], labels = new_ind1, pos = 3, cex = 0.8)
text(raw_dat1[new_ind2, ], labels = new_ind2, pos = 3, cex = 0.8)
text(raw_dat1[new_ind3, ], labels = new_ind3, pos = 3, cex = 0.8)

