### Null Scenario

#Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
#Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
#Z3 = rmvnorm(n, mean= cent3, sigma = sig3)
n=100
rotmat = matrix(c(cos(pi/3), -sin(pi/3), sin(pi/3), cos(pi/3)), nrow=2, byrow=T)

X1 = 3*rmvnorm(n, mean= cent1, sigma = sig1)
X2 = 2*rmvnorm(n, mean= cent2, sigma = sig2)
X3 = rmvnorm(n, mean= cent3, sigma = sig3)

Y1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
Y2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
Y3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat

K1 = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
K2 = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
K3 = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat

raw_dat1 = rbind(X1, X2, X3)
#purterb for better visualization
set.seed(1)
purt.size = n*.2
purt.ind = sample(1:(3*n), size=3*purt.size)
raw_dat1[purt.ind[1:purt.size], ] = 3*rmvnorm(purt.size, mean= cent1, sigma = sig1)
raw_dat1[purt.ind[1:purt.size + purt.size], ] = 2*rmvnorm(purt.size, mean= cent2, sigma = sig2)
raw_dat1[purt.ind[1:purt.size + purt.size*2], ] = rmvnorm(purt.size, mean= cent3, sigma = sig3)

raw_dat2 = rbind(Y1, Y2, Y3)
raw_dat3 = rbind(K1, K2, K3)

layout(matrix(1:6, nrow=2, byrow=T))
plot(raw_dat1)
#points(raw_dat1[1:n, ], col=2)
#points(raw_dat1[(n+1):(2*n), ], col=3)
#points(raw_dat1[(2*n+1):(3*n), ], col=4)


plot(raw_dat2)
points(raw_dat2[1:n, ], col="red")
points(raw_dat2[(n+1):(2*n), ], col="pink")
points(raw_dat2[(2*n+1):(3*n), ], col="darkred")

purt.size = n*.2
indset = 1:(3*n)
purt.ind2 = sample(1:(3*n), size=3*purt.size)


ind1 = union( setdiff(1:n, purt.ind2), purt.ind2[1:purt.size])
ind2 = union( setdiff((n+1):(2*n), purt.ind2), purt.ind2[1:purt.size + purt.size])
ind3 = union(setdiff((2*n+1):(3*n), purt.ind2), purt.ind2[1:purt.size + 2*purt.size])



raw_dat3[ind1, ] = rmvnorm(length(ind1), mean= cent1, sigma = sig1)%*%rotmat
raw_dat3[ind2, ] = rmvnorm(length(ind2), mean= cent2, sigma = sig2)%*%rotmat
raw_dat3[ind3, ] = rmvnorm(length(ind3), mean= cent3, sigma = sig3)%*%rotmat

plot(raw_dat3) 

points(raw_dat3[ind1, ], col="green")
points(raw_dat3[ind2, ], col="lightgreen")
points(raw_dat3[ind3, ], col="darkgreen")


plot(raw_dat1)
plot(raw_dat1)
points(raw_dat1[1:n, ], col="red")
points(raw_dat1[(n+1):(2*n), ], col="pink")
points(raw_dat1[(2*n+1):(3*n), ], col="darkred")

plot(raw_dat1)
points(raw_dat1[ind1, ], col="green")
points(raw_dat1[ind2, ], col="lightgreen")
points(raw_dat1[ind3, ], col="darkgreen")

### Alt. Scenario

### Null Scenario

#Z1 = rmvnorm(n, mean= cent1, sigma = sig1)
#Z2 = rmvnorm(n, mean= cent2, sigma = sig2)
#Z3 = rmvnorm(n, mean= cent3, sigma = sig3)

layout(matrix(1:6, nrow=2, byrow=T))
plot(raw_dat1)

plot(raw_dat2)
points(raw_dat2[1:n, ], col="red")
points(raw_dat2[(n+1):(2*n), ], col="pink")
points(raw_dat2[(2*n+1):(3*n), ], col="darkred")

new_ind = sample(1:(3*n))

new_ind1 = new_ind[1:n]
new_ind2 = new_ind[1:n+n]
new_ind3 = new_ind[1:n+2*n]

raw_dat3_new = raw_dat3
raw_dat3_new[new_ind1, ] = rmvnorm(n, mean= cent1, sigma = sig1)%*%rotmat
raw_dat3_new[new_ind2, ] = rmvnorm(n, mean= cent2, sigma = sig2)%*%rotmat
raw_dat3_new[new_ind3, ] = rmvnorm(n, mean= cent3, sigma = sig3)%*%rotmat


plot(raw_dat3_new) 

points(raw_dat3_new[new_ind1, ], col="green")
points(raw_dat3_new[new_ind2, ], col="lightgreen")
points(raw_dat3_new[new_ind3, ], col="darkgreen")



plot(raw_dat1)
plot(raw_dat1)
points(raw_dat1[1:n, ], col="red")
points(raw_dat1[(n+1):(2*n), ], col="pink")
points(raw_dat1[(2*n+1):(3*n), ], col="darkred")

plot(raw_dat1)
points(raw_dat1[new_ind1, ], col="green")
points(raw_dat1[new_ind2, ], col="lightgreen")
points(raw_dat1[new_ind3, ], col="darkgreen")

