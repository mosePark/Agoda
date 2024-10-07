'
2024-10-07
data1에 쓸 시뮬레이션 데이터 생성
'

# Load libraries
library(MASS)
library(ggplot2)

# Setting seed
set.seed(241007)

# Define centroids (mean vectors)
mu1 <- c(2, 2)
mu2 <- c(-2, 2)
mu3 <- c(0, -3)

# Adjusted covariance matrices for more overlap
cov1 <- matrix(c(1.5, 0.6, 0.6, 1.5), nrow = 2)
cov2 <- matrix(c(1.5, -0.4, -0.4, 1.5), nrow = 2)
cov3 <- matrix(c(1.5, 0.3, 0.3, 1.5), nrow = 2)

# Sampling 100 data points ~ MultiNormal dist for each centroid
cluster1 <- mvrnorm(100, mu = mu1, Sigma = cov1)
cluster2 <- mvrnorm(100, mu = mu2, Sigma = cov2)
cluster3 <- mvrnorm(100, mu = mu3, Sigma = cov3)

# Combine the data into a single data frame for ggplot
dat1 <- data.frame(
  X1 = c(cluster1[,1], cluster2[,1], cluster3[,1]),
  X2 = c(cluster1[,2], cluster2[,2], cluster3[,2]),
  Cluster = factor(rep(1:3, each = 100))
)

# Define colors for clusters
cluster_colors <- c("orange", "green", "blue")

# Create the ggplot
p <- ggplot(dat1, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
  # stat_ellipse(type = "norm", linetype = "dashed", size = 1) +  # Add ellipses
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Multivariate Normal Distribution with Cluster Overlap") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-6, 6), ylim = c(-6, 6))

p

# Save the plot with high DPI
ggsave("cluster_plot.png", plot = p, width = 8, height = 6, dpi = 1000)
