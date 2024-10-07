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


# Define linear transformation parameters for each cluster
a1 <- 1.2; b1 <- c(3, -2)   # Transformation for Cluster 1
a2 <- 1.1; b2 <- c(-3, 2)   # Transformation for Cluster 2
a3 <- 0.9; b3 <- c(2, 3)    # Transformation for Cluster 3

# Apply linear transformation to each cluster
cluster1_transformed <- a1 * cluster1 + matrix(rep(b1, each = 100), ncol = 2, byrow = TRUE)
cluster2_transformed <- a2 * cluster2 + matrix(rep(b2, each = 100), ncol = 2, byrow = TRUE)
cluster3_transformed <- a3 * cluster3 + matrix(rep(b3, each = 100), ncol = 2, byrow = TRUE)

# Combine transformed data into a new data frame for plotting
dat2 <- data.frame(
  X1 = c(cluster1_transformed[,1], cluster2_transformed[,1], cluster3_transformed[,1]),
  X2 = c(cluster1_transformed[,2], cluster2_transformed[,2], cluster3_transformed[,2]),
  Cluster = factor(rep(1:3, each = 100))
)

# Define colors for clusters
cluster_colors <- c("orange", "green", "blue")

# Plot the transformed data
p2 <- ggplot(dat2, aes(x = X1, y = X2, color = Cluster)) +
  geom_point(size = 1.5) +
  scale_color_manual(values = cluster_colors) +
  labs(x = expression(X[1]), y = expression(X[2]), title = "Transformed Data with Linear Transformation (data2)") +
  theme_minimal(base_size = 15) +
  theme(
    panel.grid.major = element_line(color = "grey80", linetype = "solid"),
    panel.grid.minor = element_line(color = "grey90", linetype = "solid"),
    axis.line = element_line(color = "black"),
    legend.position = "top"
  ) +
  coord_cartesian(xlim = c(-10, 10), ylim = c(-10, 10))

p2

# Save the plot with high DPI
ggsave("cluster_plot_no_ellipses.png", plot = p, width = 8, height = 6, dpi = 1000)
