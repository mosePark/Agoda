'

NULL

'

library(mvtnorm)
library(colorspace)
library(grDevices)
library(dplyr)
library(yarrr)

## color

ylorrd_colors <- hcl.colors(5, "YlOrRd")
ylgnbu_colors <- hcl.colors(5, "YlGnBu")

## Centroid
cent1 = c(sqrt(3), 1); cent2 = c(-sqrt(3),1); cent3 = c(0,-2)

# Sigma
sig1 = matrix(c(1,.6,.6,1), nrow=2)
sig2 = matrix(c(1,-.6,-.6,1), nrow=2)
sig3 = matrix(c(sqrt(2)/2,0,0,sqrt(2)), nrow=2)


################################################################################
### Null Scenario ##############################################################
################################################################################


n = 120
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

distance_threshold <- 1.1

# Raw data2 - transparent

distances12 <- as.matrix(dist(rbind(raw_dat2[1:n, ], raw_dat2[(n + 1):(2 * n), ])))
distances13 <- as.matrix(dist(rbind(raw_dat2[1:n, ], raw_dat2[(2 * n + 1):(3 * n), ])))
distances23 <- as.matrix(dist(rbind(raw_dat2[(n + 1):(2 * n), ], raw_dat2[(2 * n + 1):(3 * n), ])))


overlap_12 <- which(distances12[1:n, (n + 1):(2 * n)] < distance_threshold, arr.ind = TRUE)
overlap_13 <- which(distances13[1:n, (n + 1):(2 * n)] < distance_threshold, arr.ind = TRUE)
overlap_23 <- which(distances23[1:n, (n + 1):(2 * n)] < distance_threshold, arr.ind = TRUE)

overlap_indices_1 <- intersect(overlap_12[, 1], overlap_13[, 1])
overlap_indices_2 <- intersect(overlap_12[, 2], overlap_23[, 1]) + n
overlap_indices_3 <- intersect(overlap_13[, 2], overlap_23[, 2]) + 2 * n


overlap_indices_1 <- unique(overlap_indices_1)
overlap_indices_2 <- unique(overlap_indices_2)
overlap_indices_3 <- unique(overlap_indices_3)


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

# Set a distance threshold for identifying overlaps
distance_threshold <- 1.1

# Calculate pairwise distances between clusters in raw_dat3
distances12_3 <- as.matrix(dist(rbind(raw_dat3[ind1, ], raw_dat3[ind2, ])))
distances13_3 <- as.matrix(dist(rbind(raw_dat3[ind1, ], raw_dat3[ind3, ])))
distances23_3 <- as.matrix(dist(rbind(raw_dat3[ind2, ], raw_dat3[ind3, ])))

# Identify overlapping indices where distances are below the threshold
overlap_12_3 <- which(distances12_3[1:length(ind1), (length(ind1) + 1):(length(ind1) + length(ind2))] < distance_threshold, arr.ind = TRUE)
overlap_13_3 <- which(distances13_3[1:length(ind1), (length(ind1) + 1):(length(ind1) + length(ind3))] < distance_threshold, arr.ind = TRUE)
overlap_23_3 <- which(distances23_3[1:length(ind2), (length(ind2) + 1):(length(ind2) + length(ind3))] < distance_threshold, arr.ind = TRUE)

# Extract indices that appear in all three overlaps for raw_dat3
overlap_indices_1_3 <- ind1[intersect(overlap_12_3[, 1], overlap_13_3[, 1])]
overlap_indices_2_3 <- ind2[intersect(overlap_12_3[, 2], overlap_23_3[, 1])]
overlap_indices_3_3 <- ind3[intersect(overlap_13_3[, 2], overlap_23_3[, 2])]

# Ensure indices are unique
overlap_indices_1_3 <- unique(overlap_indices_1_3)
overlap_indices_2_3 <- unique(overlap_indices_2_3)
overlap_indices_3_3 <- unique(overlap_indices_3_3)


# Centroid 계산

K = 3


dat2_on_dat1_cent <- matrix(NA, nrow = K, ncol = ncol(raw_dat1))
dat3_on_dat1_cent <- matrix(NA, nrow = K, ncol = ncol(raw_dat1))

centroids <- list(cent1, cent2, cent3)

for (k in 1:K) {
  
  distances_dat2 <- apply(raw_dat2, 1, function(x) sum((x - centroids[[k]])^2))
  distances_dat3 <- apply(raw_dat3, 1, function(x) sum((x - centroids[[k]])^2))
  
  closest_points_dat2 <- which(distances_dat2 == min(distances_dat2))
  closest_points_dat3 <- which(distances_dat3 == min(distances_dat3))
  
  dat2_on_dat1_cent[k, ] <- colMeans(raw_dat2[closest_points_dat2, , drop = FALSE])
  dat3_on_dat1_cent[k, ] <- colMeans(raw_dat3[closest_points_dat3, , drop = FALSE])
}

# abc 포인트

a <- raw_dat1[8, ]             # Cluster 1 대표 포인트
b <- raw_dat1[n + 5, ]         # Cluster 2 대표 포인트
c <- raw_dat1[2 * n + 5, ]     # Cluster 3 대표 포인트  (변경된 위치)

a_2 <- raw_dat2[8, ]            # raw_dat2에서 변환된 a의 위치
b_2 <- raw_dat2[n + 5, ]        # raw_dat2에서 변환된 b의 위치
c_2 <- raw_dat2[2 * n + 5, ]    # raw_dat2에서 변환된 c의 위치 (변경된 위치)

a_3 <- raw_dat3[ind1[8], ]      # raw_dat3에서 변환된 a의 위치 (ind1에서 첫 번째)
b_3 <- raw_dat3[ind2[5], ]      # raw_dat3에서 변환된 b의 위치 (ind2에서 첫 번째)
c_3 <- raw_dat3[ind3[5], ]      # raw_dat3에서 변환된 c의 위치 (변경된 위치)




################################################################################
# NULL 그리기 ##################################################################
################################################################################

# # Define point styles (use star and increase size)
# point_style <- list(pch = 21, cex = 5, col = "purple", bg = "yellow")



png("1.png", width = 2800, height = 1500)

par(oma = c(3, 3, 1, 1))
par(mfrow = c(1, 1))

par(mar = c(5.3, 4.3, 4.3, 2.3))

# Set the label size
label_size <- 1.5  # Adjust this value to make labels larger
label_font <- 1.5    # Bold font

# 1 by 1
par(fig = c(0, 0.33, 0.5, 1), new = TRUE)
plot(raw_dat2,, xlab='', ylab='', cex=5, xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font)
points(raw_dat2[setdiff(1:n, overlap_indices_1), ], pch=21, cex=5, col=yarrr::transparent("#7D0025", trans.val = 0.5), bg=yarrr::transparent("#7D0025", trans.val = 0.5))
points(raw_dat2[setdiff((n + 1):(2 * n), overlap_indices_2), ],pch=21, cex=5, col=yarrr::transparent("#DA3500", trans.val = 0.5), bg=yarrr::transparent("#DA3500", trans.val = 0.5))
points(raw_dat2[setdiff((2 * n + 1):(3 * n), overlap_indices_3), ],pch=21, cex=5, col=yarrr::transparent("#F39300", trans.val = 0.5), bg=yarrr::transparent("#F39300", trans.val = 0.5))

points(raw_dat2[overlap_indices_1, ], pch=21, cex=5, col=yarrr::transparent("#7D0025", trans.val = 0.5), bg=yarrr::transparent("#7D0025", trans.val = 0.8))
points(raw_dat2[overlap_indices_2, ], pch=21, cex=5, col=yarrr::transparent("#DA3500", trans.val = 0.5), bg=yarrr::transparent("#DA3500", trans.val = 0.8))
points(raw_dat2[overlap_indices_3, ], pch=21, cex=5, col=yarrr::transparent("#F39300", trans.val = 0.5), bg=yarrr::transparent("#F39300", trans.val = 0.8))

points(a_2[1], a_2[2], pch = 21, cex = 5, col = "green", bg = "#7D0025")
points(b_2[1], b_2[2], pch = 21, cex = 5, col = "green", bg = "#DA3500")
points(c_2[1], c_2[2], pch = 21, cex = 5, col = "green", bg = "#F39300")

text(a_2[1], a_2[2], labels = "a", pos = 3, cex = 5, col = "black")
text(b_2[1], b_2[2], labels = "b", pos = 3, cex = 5, col = "black")
text(c_2[1], c_2[2], labels = "c", pos = 3, cex = 5, col = "black")



# 1 by 2
par(fig = c(0.33, 0.66, 0.5, 1), new = TRUE)
plot(raw_dat1, cex=5, xlab='', ylab='', xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font)

points(a[1], a[2], pch = 21, cex = 5, bg = "green")
points(b[1], b[2], pch = 21, cex = 5, bg = "green")
points(c[1], c[2], pch = 21, cex = 5, bg = "green")

text(a[1], a[2], labels = "a", pos = 3, cex = 5, col = "black")
text(b[1], b[2], labels = "b", pos = 3, cex = 5, col = "black")
text(c[1], c[2], labels = "c", pos = 3, cex = 5, col = "black")



# 1 by 3
par(fig = c(0.66, 1, 0.5, 1), new = TRUE)
plot(raw_dat3, xlab ='', ylab = '',cex=5 , xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font)

points(raw_dat3[setdiff(ind1, overlap_indices_1_3), ], pch=21, cex=5, col=yarrr::transparent("#26185F", trans.val = 0.5), bg=yarrr::transparent("#26185F", trans.val = 0.5))
points(raw_dat3[setdiff(ind2, overlap_indices_2_3), ], pch=21, cex=5, col=yarrr::transparent("#007EB3", trans.val = 0.5), bg=yarrr::transparent("#007EB3", trans.val = 0.5))
points(raw_dat3[setdiff(ind3, overlap_indices_3_3), ], pch=21, cex=5, col=yarrr::transparent("#18BDB0", trans.val = 0.5), bg=yarrr::transparent("#18BDB0", trans.val = 0.5))

points(raw_dat3[overlap_indices_1_3, ], pch=21, cex=5, col=adjustcolor("#26185F", alpha.f = 0.5), bg=adjustcolor("#26185F", alpha.f = 0.5))
points(raw_dat3[overlap_indices_2_3, ], pch=21, cex=5, col=adjustcolor("#007EB3", alpha.f = 0.5), bg=adjustcolor("#007EB3", alpha.f = 0.5))
points(raw_dat3[overlap_indices_3_3, ], pch=21, cex=5, col=adjustcolor("#18BDB0", alpha.f = 0.5), bg=adjustcolor("#18BDB0", alpha.f = 0.5))

points(a_3[1], a_3[2], pch = 21, cex = 5, col = "green", bg = "#26185F")
points(b_3[1], b_3[2], pch = 21, cex = 5, col = "green", bg = "#007EB3")
points(c_3[1], c_3[2], pch = 21, cex = 5, col = "green", bg = "#18BDB0")

text(a_3[1], a_3[2], labels = "a", pos = 3, cex = 5, col = "black")
text(b_3[1], b_3[2], labels = "b", pos = 3, cex = 5, col = "black")
text(c_3[1], c_3[2], labels = "c", pos = 3, cex = 5, col = "black")


# 2 by 1
par(fig = c(0.2, 0.5, 0, 0.5), new = TRUE)
plot(raw_dat1, cex=5,xlab ='', ylab = '', xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font)

points(raw_dat1[1:n, ], pch=21, cex=5, col=yarrr::transparent("#7D0025", trans.val = 0.5), bg=yarrr::transparent("#7D0025", trans.val = 0.8))
points(raw_dat1[(n + 1):(2 * n), ], pch=21, cex=5, col=yarrr::transparent("#DA3500", trans.val = 0.5), bg=yarrr::transparent("#DA3500", trans.val = 0.8))
points(raw_dat1[(2 * n + 1):(3 * n), ], pch=21, cex=5, col=yarrr::transparent("#F39300", trans.val = 0.5), bg=yarrr::transparent("#F39300", trans.val = 0.8))

points(dat2_on_dat1_cent[1,1], dat2_on_dat1_cent[1,2], bg = "#7D0025", pch = 24, cex = 5, col='green')
points(dat2_on_dat1_cent[2,1], dat2_on_dat1_cent[2,2], bg = "#DA3500", pch = 24, cex = 5, col='green')
points(dat2_on_dat1_cent[3,1], dat2_on_dat1_cent[3,2], bg = "#F39300", pch = 24, cex = 5, col='green')



# 2 by 2
par(fig = c(0.5, 0.8, 0, 0.5), new = TRUE)
plot(raw_dat1, cex=5, xlab ='', ylab = '', xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font)

points(raw_dat1[ind1, ], pch=21, cex=5, col=yarrr::transparent("#26185F", trans.val = 0.5), bg=yarrr::transparent("#26185F", trans.val = 0.8))
points(raw_dat1[ind2, ], pch=21, cex=5, col=yarrr::transparent("#007EB3", trans.val = 0.5), bg=yarrr::transparent("#007EB3", trans.val = 0.8))
points(raw_dat1[ind3, ], pch=21, cex=5, col=yarrr::transparent("#18BDB0", trans.val = 0.5), bg=yarrr::transparent("#18BDB0", trans.val = 0.8))

points(dat2_on_dat1_cent[1,1], dat2_on_dat1_cent[1,2], bg = "#26185F", pch = 24, cex = 5, col='green')
points(dat2_on_dat1_cent[2,1], dat2_on_dat1_cent[2,2], bg = "#007EB3", pch = 24, cex = 5, col='green')
points(dat2_on_dat1_cent[3,1], dat2_on_dat1_cent[3,2], bg = "#18BDB0", pch = 24, cex = 5, col='green')

dev.off()

################################################################################
# kmeans cl#####################################################################
################################################################################
library(fossil)

dat2cl = kmeans(raw_dat2, centers= 3)
dat3cl = kmeans(raw_dat3, centers= 3)

ri_result <- rand.index(dat2cl$cluster, dat3cl$cluster)
print(ri_result)

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


plot(raw_dat3_new)
points(raw_dat3_new[new_ind1, ], col=1)
points(raw_dat3_new[new_ind2, ], col=2)
points(raw_dat3_new[new_ind3, ], col=3)

# Set a distance threshold for identifying overlaps
distance_threshold <- 1.1

# Calculate pairwise distances between clusters in raw_dat3
distances12_3 <- as.matrix(dist(rbind(raw_dat3_new[ind1, ], raw_dat3[ind2, ])))
distances13_3 <- as.matrix(dist(rbind(raw_dat3_new[ind1, ], raw_dat3[ind3, ])))
distances23_3 <- as.matrix(dist(rbind(raw_dat3_new[ind2, ], raw_dat3[ind3, ])))

# Identify overlapping indices where distances are below the threshold
overlap_12_3 <- which(distances12_3[1:length(ind1), (length(ind1) + 1):(length(ind1) + length(ind2))] < distance_threshold, arr.ind = TRUE)
overlap_13_3 <- which(distances13_3[1:length(ind1), (length(ind1) + 1):(length(ind1) + length(ind3))] < distance_threshold, arr.ind = TRUE)
overlap_23_3 <- which(distances23_3[1:length(ind2), (length(ind2) + 1):(length(ind2) + length(ind3))] < distance_threshold, arr.ind = TRUE)

# Extract indices that appear in all three overlaps for raw_dat3
overlap_indices_1_3 <- new_ind1[intersect(overlap_12_3[, 1], overlap_13_3[, 1])]
overlap_indices_2_3 <- new_ind2[intersect(overlap_12_3[, 2], overlap_23_3[, 1])]
overlap_indices_3_3 <- new_ind3[intersect(overlap_13_3[, 2], overlap_23_3[, 2])]

# Ensure indices are unique
overlap_indices_1_3 <- unique(overlap_indices_1_3)
overlap_indices_2_3 <- unique(overlap_indices_2_3)
overlap_indices_3_3 <- unique(overlap_indices_3_3)


# Centroid 계산

K = 3


dat2_on_dat1_cent <- matrix(NA, nrow = K, ncol = ncol(raw_dat1))
dat3_on_dat1_cent <- matrix(NA, nrow = K, ncol = ncol(raw_dat1))

centroids <- list(cent1, cent2, cent3)
#
k=1
dat2_on_dat1_cent[k, ] <- colMeans(raw_dat1[union(1:n, overlap_indices_1), ])
dat3_on_dat1_cent[k, ] <- colMeans(raw_dat1[union(new_ind1, overlap_indices_1_3), ])
#
k=2
dat2_on_dat1_cent[k, ] <- colMeans(raw_dat1[union( (1:n+n), overlap_indices_2), ])
dat3_on_dat1_cent[k, ] <- colMeans(raw_dat1[na.omit(union(new_ind2, overlap_indices_2_3)), ])
#
k=3
dat2_on_dat1_cent[k, ] <- colMeans(raw_dat1[union( (1:n+2*n), overlap_indices_3), ])
dat3_on_dat1_cent[k, ] <- colMeans(raw_dat1[union(new_ind3, overlap_indices_3_3), ])


#for (k in 1:K) {
#  
#  distances_dat2 <- apply(raw_dat2, 1, function(x) sum((x - centroids[[k]])^2))
#  distances_dat3 <- apply(raw_dat3_new, 1, function(x) sum((x - centroids[[k]])^2))
#  
#  closest_points_dat2 <- which(distances_dat2 == min(distances_dat2))
#  closest_points_dat3 <- which(distances_dat3 == min(distances_dat3))
#  
#  dat2_on_dat1_cent[k, ] <- colMeans(raw_dat1[closest_points_dat2, , drop = FALSE])
#  dat3_on_dat1_cent[k, ] <- colMeans(raw_dat1[closest_points_dat3, , drop = FALSE])
#}

# Select representative points in raw_dat1 for each cluster
i=40
j = 17
a <- raw_dat1[new_ind1[i], ]             # Cluster 1 대표 포인트
b <- raw_dat1[new_ind2[j], ]         # Cluster 2 대표 포인트
c <- raw_dat1[new_ind3[5], ]     # Cluster 3 대표 포인트 (변경된 위치)

# Find the corresponding transformed points in raw_dat2 and raw_dat3_new
a_2 <- raw_dat2[new_ind1[i], ]            # raw_dat2에서 변환된 a의 위치
b_2 <- raw_dat2[new_ind2[j], ]        # raw_dat2에서 변환된 b의 위치
c_2 <- raw_dat2[new_ind3[5], ]    # raw_dat2에서 변환된 c의 위치 (변경된 위치)

a_3 <- raw_dat3_new[new_ind1[i], ]  # raw_dat3_new에서 변환된 a의 위치
b_3 <- raw_dat3_new[new_ind2[j], ]  # raw_dat3_new에서 변환된 b의 위치
c_3 <- raw_dat3_new[new_ind3[5], ]  # raw_dat3_new에서 변환된 c의 위치 (변경된 위치)


################################################################################
# Alter 그리기 #################################################################
################################################################################

png("2.png", width = 2800, height = 1500)

par(oma = c(3, 3, 1, 1))
par(mfrow = c(1, 1))

par(mar = c(5.3, 4.3, 4.3, 2.3))


# Set the label size
label_size <- 1.55  # Adjust this value to make labels larger
label_font <- 1.5    # Bold font



# 1 by 1
par(fig = c(0, 0.33, 0.5, 1), new = TRUE)
plot(raw_dat2, xlab='', ylab='', cex=5, xaxt='n', yaxt='n', ylim = c(-5,6), xlim=c(-5,5), cex.lab = label_size, font.lab = label_font, col="white")

points(raw_dat2[setdiff(1:n, overlap_indices_1), ], pch=21, cex=4, col=yarrr::transparent("#1b9e77", trans.val = 0.5), bg=yarrr::transparent("#1b9e77", trans.val = 0.5))
points(raw_dat2[setdiff((n + 1):(2 * n), overlap_indices_2), ],pch=21, cex=4, col=yarrr::transparent("#d95f02", trans.val = 0.5), bg=yarrr::transparent("#d95f02", trans.val = 0.5))
points(raw_dat2[setdiff((2 * n + 1):(3 * n), overlap_indices_3), ],pch=21, cex=4, col=yarrr::transparent("#7570b3", trans.val = 0.5), bg=yarrr::transparent("#7570b3", trans.val = 0.5))

points(raw_dat2[overlap_indices_1, ], pch=21, cex=4, col=yarrr::transparent("#1b9e77", trans.val = 0.5), bg=yarrr::transparent("#1b9e77", trans.val = 0.8))
points(raw_dat2[overlap_indices_2, ], pch=21, cex=4, col=yarrr::transparent("#d95f02", trans.val = 0.5), bg=yarrr::transparent("#d95f02", trans.val = 0.8))
points(raw_dat2[overlap_indices_3, ], pch=21, cex=4, col=yarrr::transparent("#7570b3", trans.val = 0.5), bg=yarrr::transparent("#7570b3", trans.val = 0.8))

points(a_2[1], a_2[2], pch = 16, cex = 8, col = "yellow", bg = "#1b9e77")
points(b_2[1], b_2[2], pch = 16, cex = 8, col = "yellow", bg = "#d95f02")
points(c_2[1], c_2[2], pch = 16, cex = 8, col = "yellow", bg = "#7570b3")

points(a_2[1], a_2[2], pch = 16, cex = 6, col = "#1b9e77")
points(b_2[1], b_2[2], pch = 16, cex = 6, col = "#d95f02")
points(c_2[1], c_2[2], pch = 16, cex = 6, col = "#7570b3")


#text(a_2[1], a_2[2], labels = "a", pos = 3, cex = 5, col = "black")
#text(b_2[1], b_2[2], labels = "b", pos = 3, cex = 5, col = "black")
#text(c_2[1], c_2[2], labels = "c", pos = 3, cex = 5, col = "black")



# 1 by 2
par(fig = c(0.33, 0.66, 0.5, 1), new = TRUE)
plot(raw_dat1, cex=4, xlab='', ylab='', xaxt='n', yaxt='n', xlim = c(-11,11), ylim =c(-7, 11), cex.lab = label_size, font.lab = label_font, col=yarrr::transparent("darkgrey", trans.val = 0.5), bg=yarrr::transparent("darkgrey", trans.val = 0.5), pch=16)

points(a[1], a[2], pch = 16, cex = 8, col = "green")
points(b[1], b[2], pch = 16, cex = 8, col = "green")
points(c[1], c[2], pch = 16, cex = 8, col = "green")

points(a[1], a[2], pch = 16, cex = 6, col = "black")
points(b[1], b[2], pch = 16, cex = 6, col = "black")
points(c[1], c[2], pch = 16, cex = 6, col = "black")
#text(a[1], a[2], labels = "a", pos = 3, cex = 5, col = "black")
#text(b[1], b[2], labels = "b", pos = 3, cex = 5, col = "black")
#text(c[1], c[2], labels = "c", pos = 3, cex = 5, col = "black")



# 1 by 3 (new)
par(fig = c(0.66, 1, 0.5, 1), new = TRUE)
plot(raw_dat3_new, cex=5,xlab='', ylab='', xaxt='n', yaxt='n', xlim = c(-5,7), ylim = c(-4,6), cex.lab = label_size, font.lab = label_font, col="white")

points(raw_dat3_new[setdiff(new_ind1, overlap_indices_1_3), ], pch=21, cex=4, col=yarrr::transparent("#1b5faa", trans.val = 0.5), bg=yarrr::transparent("#1b5faa", trans.val = 0.5))
points(raw_dat3_new[setdiff(new_ind2, overlap_indices_2_3), ], pch=21, cex=4, col=yarrr::transparent("#90cef1", trans.val = 0.5), bg=yarrr::transparent("#90cef1", trans.val = 0.5))
points(raw_dat3_new[setdiff(new_ind3, overlap_indices_3_3), ], pch=21, cex=4, col=yarrr::transparent("#ffc20e", trans.val = 0.5), bg=yarrr::transparent("#ffc20e", trans.val = 0.5))

points(raw_dat3_new[overlap_indices_1_3, ], pch=21, cex=4, col=adjustcolor("#1b5faa", alpha.f = 0.5), bg=adjustcolor("#1b5faa", alpha.f = 0.5))
points(raw_dat3_new[overlap_indices_2_3, ], pch=21, cex=4, col=adjustcolor("#90cef1", alpha.f = 0.5), bg=adjustcolor("#90cef1", alpha.f = 0.5))
points(raw_dat3_new[overlap_indices_3_3, ], pch=21, cex=4, col=adjustcolor("#ffc20e", alpha.f = 0.5), bg=adjustcolor("#ffc20e", alpha.f = 0.5))

points(a_3[1], a_3[2], pch = 16, cex = 8, col = "red")
points(b_3[1], b_3[2], pch = 16, cex = 8, col = "red")
points(c_3[1], c_3[2], pch = 16, cex = 8, col = "red")

points(a_3[1], a_3[2], pch = 16, cex = 6, col =  "#1b5faa")
points(b_3[1], b_3[2], pch = 16, cex = 6, col =  "#90cef1")
points(c_3[1], c_3[2], pch = 16, cex = 6, col =  "#ffc20e")

#text(a_3[1], a_3[2], labels = "a", pos = 3, cex = 5, col = "black")
#text(b_3[1], b_3[2], labels = "b", pos = 3, cex = 5, col = "black")
#text(c_3[1], c_3[2], labels = "c", pos = 3, cex = 5, col = "black")


# 2 by 1
par(fig = c(0.2, 0.5, 0, 0.5), new = TRUE)
plot(raw_dat1, cex=5, xlab ='', ylab = '', xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font, col="white")

points(raw_dat1[1:n, ], pch=21, cex=4, col=yarrr::transparent("#1b9e77", trans.val = 0.5), bg=yarrr::transparent("#1b9e77", trans.val = 0.5))
points(raw_dat1[(n + 1):(2 * n), ], pch=21, cex=4, col=yarrr::transparent("#d95f02", trans.val = 0.5), bg=yarrr::transparent("#d95f02", trans.val = 0.5))
points(raw_dat1[(2 * n + 1):(3 * n), ], pch=21, cex=4, col=yarrr::transparent("#7570b3", trans.val = 0.5), bg=yarrr::transparent("#7570b3", trans.val = 0.5))

points(dat2_on_dat1_cent[1,1], dat2_on_dat1_cent[1,2],  pch = 18, cex = 8, col='yellow')
points(dat2_on_dat1_cent[2,1], dat2_on_dat1_cent[2,2],  pch = 18, cex = 8, col='yellow')
points(dat2_on_dat1_cent[3,1], dat2_on_dat1_cent[3,2],  pch = 18, cex = 8, col='yellow')

points(dat2_on_dat1_cent[1,1], dat2_on_dat1_cent[1,2], bg = "#1b9e77", pch = "+", cex = 6, col="#1b9e77")
points(dat2_on_dat1_cent[2,1], dat2_on_dat1_cent[2,2], bg = "#d95f02", pch = "+", cex = 6, col="#d95f02")
points(dat2_on_dat1_cent[3,1], dat2_on_dat1_cent[3,2], bg = "#7570b3", pch = "+", cex = 6, col="#7570b3")

points(a[1], a[2], pch = 16, cex = 8, col = "yellow")
points(b[1], b[2], pch = 16, cex = 8, col = "yellow")
points(c[1], c[2], pch = 16, cex = 8, col = "yellow")

points(a[1], a[2], pch = 16, cex = 6, col = "#1b9e77")
points(b[1], b[2], pch = 16, cex = 6, col = "#d95f02")
points(c[1], c[2], pch = 16, cex = 6, col = "#7570b3")

# 2 by 2
par(fig = c(0.5, 0.8, 0, 0.5), new = TRUE)
plot(raw_dat1, cex=5, xlab ='', ylab = '', xlim=c(-6, 6), ylim=c(-6, 6), xaxt='n', yaxt='n', cex.lab = label_size, font.lab = label_font, col="white")

points(raw_dat1[new_ind1, ], pch=21, cex=4, col=yarrr::transparent("#1b5faa", trans.val = 0.5), bg=yarrr::transparent("#1b5faa", trans.val = 0.5))
points(raw_dat1[new_ind2, ], pch=21, cex=4, col=yarrr::transparent("#90cef1", trans.val = 0.5), bg=yarrr::transparent("#90cef1", trans.val = 0.5))
points(raw_dat1[new_ind3, ], pch=21, cex=4, col=yarrr::transparent("#ffc20e", trans.val = 0.5), bg=yarrr::transparent("#ffc20e", trans.val = 0.5))

points(dat3_on_dat1_cent[1,1], dat3_on_dat1_cent[1,2],  pch = 18, cex = 8, col='red')
points(dat3_on_dat1_cent[2,1], dat3_on_dat1_cent[2,2],  pch = 18, cex = 8, col='red')
points(dat3_on_dat1_cent[3,1], dat3_on_dat1_cent[3,2],  pch = 18, cex = 8, col='red')

points(dat3_on_dat1_cent[1,1], dat3_on_dat1_cent[1,2], bg = "#1b5faa", pch = "+", cex = 6, col="#1b5faa")
points(dat3_on_dat1_cent[2,1], dat3_on_dat1_cent[2,2], bg = "#90cef1", pch = "+", cex = 6, col="#90cef1")
points(dat3_on_dat1_cent[3,1], dat3_on_dat1_cent[3,2], bg = "#ffc20e", pch = "+", cex = 6, col="#ffc20e")

points(a[1], a[2], pch = 16, cex = 8, col = "red")
points(b[1], b[2], pch = 16, cex = 8, col = "red")
points(c[1], c[2], pch = 16, cex = 8, col = "red")

points(a[1], a[2], pch = 16, cex = 6, col = "#1b5faa")
points(b[1], b[2], pch = 16, cex = 6, col = "#90cef1")
points(c[1], c[2], pch = 16, cex = 6, col = "#ffc20e")


dev.off()


################################################################################
# kmeans cl#####################################################################
################################################################################
library(fossil)

dat2cl_a = kmeans(raw_dat2, centers= 3)
dat3cl_a = kmeans(raw_dat3_new, centers= 3)

ri_result <- rand.index(dat2cl_a$cluster, dat3cl_a$cluster)
print(ri_result)

