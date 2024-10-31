# 그래프 그리기

par(mfrow = c(1, 1))

par(fig = c(0, 0.33, 0.5, 1), new = TRUE)
plot()

par(fig = c(0.33, 0.66, 0.5, 1), new = TRUE)
plot(raw_dat1)

par(fig = c(0.66, 1, 0.5, 1), new = TRUE)
plot()

par(fig = c(0.2, 0.5, 0, 0.5), new = TRUE)
plot()

par(fig = c(0.5, 0.8, 0, 0.5), new = TRUE)
plot()

dev.off()
