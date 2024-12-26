options(repos = c(CRAN = 'https://cran.r-project.org'))

if (!requireNamespace("energy", quietly = TRUE)) {
  install.packages("energy", lib = "/home1/mose1103/R/library")
}

if (!requireNamespace("Ball", quietly = TRUE)) {
  install.packages("Ball", lib = "/home1/mose1103/R/library")
}

if (!requireNamespace("data.table", quietly = TRUE)) {
  install.packages("data.table", lib = "/home1/mose1103/R/library")
}

setwd("/home1/mose1103/agoda/comparison/")

make_matrix = function(xvec){
  return_mat = matrix(NA, nrow=length(xvec), ncol=1536)
  for (i in 1:length(xvec)){
    x = xvec[i]
    x_clean = gsub("\\[|\\]", "", x)
    x_numeric <- as.numeric(unlist(strsplit(x_clean, ","))) 
    return_mat[i,] = x_numeric
  }
  return(return_mat)
}


comparison_tests <- function(source = 'quora', X, Y, X_name = "X", Y_name = "Y", kfds = 5, method = c("eqdist", "ball", "kernel"), seed = 123) {
  set.seed(seed)
  idx <- which(df$source == source)
  
  X <- X[idx, , drop = FALSE]
  Y <- Y[idx, , drop = FALSE]
  
  n <- nrow(X)
  folds <- split(sample(1:n), rep(1:kfds, length.out = n))
  
  res <- list()
  
  generate_filename <- function(source, X_name, Y_name, method, seed) {
    paste0(
      "comparison_tests_",
      source, "_",
      X_name, "_",
      Y_name, "_",
      method, "_",
      seed, ".txt"
    )
  }
  
  if (TRUE) {
    output_file <- generate_filename(source, X_name, Y_name, method, seed)
  }
  
  output_content <- capture.output({
    for (i in 1:kfds) {
      current_idx <- folds[[i]]
      X_part <- X[current_idx, , drop = FALSE]
      Y_part <- Y[current_idx, , drop = FALSE]
      
      if (method == "eqdist") {
        combined_data <- rbind(X_part, Y_part)
        test_result <- energy::eqdist.etest(dist(combined_data), sizes = c(nrow(X_part), nrow(Y_part)), R = 199)
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(eqdist) ###\n")
        print(test_result)
        
      } else if (method == "ball") {
        test_result <- Ball::bd.test(X_part, Y_part, num.permutations=199)
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(ball) ###\n")
        print(test_result)
        
      } else if (method == "kernel") {
        sig <- med_sigma(X_part, Y_part)
        test_result <- kerTests::kertests(X_part, Y_part, sigma = sig)
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(kernel) ###\n")
        print(test_result)
      }
    }
  })
  
  if (TRUE) {
    writeLines(output_content, con = output_file)
    cat("Saved output to file:", output_file, "\n")
  }
  
  return(res)
}


set.seed(123)


'
Load Data
'

df <- read.csv("hug-ori.csv", header=T)

SVD_ori <- readRDS("SVD_ori.rds")

SVD_gen1_0_1 <- readRDS("SVD_gen1_0_1.rds")
SVD_gen2_0_1 <- readRDS("SVD_gen2_0_1.rds")
SVD_gen1_1_0_1 <- readRDS("SVD_gen1_1_0_1.rds")

SVD_gen1_0_4 <- readRDS("SVD_gen1_0_4.rds")
SVD_gen2_0_4 <- readRDS("SVD_gen2_0_4.rds")
SVD_gen1_1_0_4 <- readRDS("SVD_gen1_1_0_4.rds")

SVD_gen1_0_7 <- readRDS("SVD_gen1_0_7.rds")
SVD_gen2_0_7 <- readRDS("SVD_gen2_0_7.rds")
SVD_gen1_1_0_7 <- readRDS("SVD_gen1_1_0_7.rds")

SVD_gen1_1_0 <- readRDS("SVD_gen1_1_0.rds")
SVD_gen2_1_0 <- readRDS("SVD_gen2_1_0.rds")
SVD_gen1_1_1_0 <- readRDS("SVD_gen1_1_1_0.rds")

SVD_gen1_1_5 <- readRDS("SVD_gen1_1_5.rds")
SVD_gen2_1_5 <- readRDS("SVD_gen2_1_5.rds")
SVD_gen1_1_1_5 <- readRDS("SVD_gen1_1_1_5.rds")

'
reduction
'

dim = 2

# Original Data Reduction
red_or <- SVD_ori$u %*% rbind(diag(SVD_ori$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# 0.1
red_gen1_0_1 <- SVD_gen1_0_1$u %*% rbind(diag(SVD_gen1_0_1$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen2_0_1 <- SVD_gen2_0_1$u %*% rbind(diag(SVD_gen2_0_1$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_0_1 <- SVD_gen1_1_0_1$u %*% rbind(diag(SVD_gen1_1_0_1$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# 0.4
red_gen1_0_4 <- SVD_gen1_0_4$u %*% rbind(diag(SVD_gen1_0_4$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen2_0_4 <- SVD_gen2_0_4$u %*% rbind(diag(SVD_gen2_0_4$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_0_4 <- SVD_gen1_1_0_4$u %*% rbind(diag(SVD_gen1_1_0_4$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# 0.7
red_gen1_0_7 <- SVD_gen1_0_7$u %*% rbind(diag(SVD_gen1_0_7$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen2_0_7 <- SVD_gen2_0_7$u %*% rbind(diag(SVD_gen2_0_7$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_0_7 <- SVD_gen1_1_0_7$u %*% rbind(diag(SVD_gen1_1_0_7$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# 1.0
red_gen1_1_0 <- SVD_gen1_1_0$u %*% rbind(diag(SVD_gen1_1_0$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen2_1_0 <- SVD_gen2_1_0$u %*% rbind(diag(SVD_gen2_1_0$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_1_0 <- SVD_gen1_1_1_0$u %*% rbind(diag(SVD_gen1_1_1_0$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# 1.5
red_gen1_1_5 <- SVD_gen1_1_5$u %*% rbind(diag(SVD_gen1_1_5$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen2_1_5 <- SVD_gen2_1_5$u %*% rbind(diag(SVD_gen2_1_5$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))
red_gen1_1_1_5 <- SVD_gen1_1_1_5$u %*% rbind(diag(SVD_gen1_1_1_5$d[1:dim]), matrix(0, nrow=1536-dim, ncol=dim))

# cnn_news, quora, squad_2 

'
temp 0.1
'

## eqdist ################################
# quora
za = comparison_tests(source='quora', 
                      X=red_gen1_0_1, Y=red_gen2_0_1, 
                      X_name='gen1_0_1', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zb = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_0_1, 
                      X_name='or', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zc = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_0_1, 
                      X_name='or', Y_name='gen1_0_1',
                      kfds=5, method='eqdist', seed=7)

zd = comparison_tests(source='quora', 
                      X=red_gen1_0_1, Y=red_gen1_1_0_1, 
                      X_name='gen1_0_1', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)

ze = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_0_1, 
                      X_name='or', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)


# squad_2
zf = comparison_tests(source='squad_2', 
                      X=red_gen1_0_1, Y=red_gen2_0_1, 
                      X_name='gen1_0_1', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zg = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_0_1, 
                      X_name='or', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zh = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_0_1, 
                      X_name='or', Y_name='gen1_0_1',
                      kfds=5, method='eqdist', seed=7)

zi = comparison_tests(source='squad_2', 
                      X=red_gen1_0_1, Y=red_gen1_1_0_1, 
                      X_name='gen1_0_1', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)

zj = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_0_1, 
                      X_name='or', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)


# cnn_news
zk = comparison_tests(source='cnn_news', 
                      X=red_gen1_0_1, Y=red_gen2_0_1, 
                      X_name='gen1_0_1', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zl = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_0_1, 
                      X_name='or', Y_name='gen2_0_1',
                      kfds=5, method='eqdist', seed=7)

zm = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_0_1, 
                      X_name='or', Y_name='gen1_0_1',
                      kfds=5, method='eqdist', seed=7)

zn = comparison_tests(source='cnn_news', 
                      X=red_gen1_0_1, Y=red_gen1_1_0_1, 
                      X_name='gen1_0_1', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)

z0 = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_0_1, 
                      X_name='or', Y_name='gen1_1_0_1',
                      kfds=5, method='eqdist', seed=7)


## ball ################################
# quora
a = comparison_tests(source='quora', 
                     X=red_gen1_0_1, Y=red_gen2_0_1,
                     X_name='gen1_0_1', Y_name='gen2_0_1',
                     kfds=5, method='ball', seed=7)

b = comparison_tests(source='quora', 
                     X=red_or, Y=red_gen2_0_1,
                     X_name='or', Y_name='gen2_0_1',
                     kfds=5, method='ball', seed=7)

c = comparison_tests(source='quora', 
                     X=red_or, Y=red_gen1_0_1,
                     X_name='or', Y_name='gen1_0_1',
                     kfds=5, method='ball', seed=7)

fdafs = comparison_tests(source='quora', 
                         X=red_gen1_0_1, Y=red_gen1_1_0_1,
                         X_name='gen1_0_1', Y_name='gen1_1_0_1',
                         kfds=5, method='ball', seed=7)

dsacsd = comparison_tests(source='quora', 
                          X=red_or, Y=red_gen1_1_0_1,
                          X_name='or', Y_name='gen1_1_0_1',
                          kfds=5, method='ball', seed=7)


# squad_2
cvv = comparison_tests(source='squad_2', 
                       X=red_gen1_0_1, Y=red_gen2_0_1,
                       X_name='gen1_0_1', Y_name='gen2_0_1',
                       kfds=5, method='ball', seed=7)

asdac = comparison_tests(source='squad_2', 
                         X=red_or, Y=red_gen2_0_1,
                         X_name='or', Y_name='gen2_0_1',
                         kfds=5, method='ball', seed=7)

dsafsv = comparison_tests(source='squad_2', 
                          X=red_or, Y=red_gen1_0_1,
                          X_name='or', Y_name='gen1_0_1',
                          kfds=5, method='ball', seed=7)

asdfsda = comparison_tests(source='squad_2', 
                           X=red_gen1_0_1, Y=red_gen1_1_0_1,
                           X_name='gen1_0_1', Y_name='gen1_1_0_1',
                           kfds=5, method='ball', seed=7)

vzvcx = comparison_tests(source='squad_2', 
                         X=red_or, Y=red_gen1_1_0_1,
                         X_name='or', Y_name='gen1_1_0_1',
                         kfds=5, method='ball', seed=7)


# cnn_news
acsdacdcacs = comparison_tests(source='cnn_news', 
                               X=red_gen1_0_1, Y=red_gen2_0_1,
                               X_name='gen1_0_1', Y_name='gen2_0_1',
                               kfds=5, method='ball', seed=7)

zxcvxvcv = comparison_tests(source='cnn_news', 
                            X=red_or, Y=red_gen2_0_1,
                            X_name='or', Y_name='gen2_0_1',
                            kfds=5, method='ball', seed=7)

qweqwe = comparison_tests(source='cnn_news', 
                          X=red_or, Y=red_gen1_0_1,
                          X_name='or', Y_name='gen1_0_1',
                          kfds=5, method='ball', seed=7)

sdafewgdv = comparison_tests(source='cnn_news', 
                             X=red_gen1_0_1, Y=red_gen1_1_0_1,
                             X_name='gen1_0_1', Y_name='gen1_1_0_1',
                             kfds=5, method='ball', seed=7)

qwcxc = comparison_tests(source='cnn_news', 
                         X=red_or, Y=red_gen1_1_0_1,
                         X_name='or', Y_name='gen1_1_0_1',
                         kfds=5, method='ball', seed=7)



'
temp 0.4
'

## eqdist ################################
# quora
qp = comparison_tests(source='quora', 
                      X=red_gen1_0_4, Y=red_gen2_0_4,
                      X_name='gen1_0_4', Y_name='gen2_0_4',
                      kfds=5, method='eqdist', seed=7)

qeqdcv = comparison_tests(source='quora', 
                          X=red_or, Y=red_gen2_0_4,
                          X_name='or', Y_name='gen2_0_4',
                          kfds=5, method='eqdist', seed=7)

qwewqsa = comparison_tests(source='quora', 
                           X=red_or, Y=red_gen1_0_4,
                           X_name='or', Y_name='gen1_0_4',
                           kfds=5, method='eqdist', seed=7)

abcd = comparison_tests(source='quora', 
                        X=red_gen1_0_4, Y=red_gen1_1_0_4,
                        X_name='gen1_0_4', Y_name='gen1_1_0_4',
                        kfds=5, method='eqdist', seed=7)

efg = comparison_tests(source='quora', 
                       X=red_or, Y=red_gen1_1_0_4,
                       X_name='or', Y_name='gen1_1_0_4',
                       kfds=5, method='eqdist', seed=7)


# squad_2
hic = comparison_tests(source='squad_2', 
                       X=red_gen1_0_4, Y=red_gen2_0_4,
                       X_name='gen1_0_4', Y_name='gen2_0_4',
                       kfds=5, method='eqdist', seed=7)

klml = comparison_tests(source='squad_2', 
                        X=red_or, Y=red_gen2_0_4,
                        X_name='or', Y_name='gen2_0_4',
                        kfds=5, method='eqdist', seed=7)

qsc = comparison_tests(source='squad_2', 
                       X=red_or, Y=red_gen1_0_4,
                       X_name='or', Y_name='gen1_0_4',
                       kfds=5, method='eqdist', seed=7)

zxpd = comparison_tests(source='squad_2', 
                        X=red_gen1_0_4, Y=red_gen1_1_0_4,
                        X_name='gen1_0_4', Y_name='gen1_1_0_4',
                        kfds=5, method='eqdist', seed=7)

afdsdfdfvbn = comparison_tests(source='squad_2', 
                               X=red_or, Y=red_gen1_1_0_4,
                               X_name='or', Y_name='gen1_1_0_4',
                               kfds=5, method='eqdist', seed=7)


# cnn_news
agj = comparison_tests(source='cnn_news', 
                       X=red_gen1_0_4, Y=red_gen2_0_4,
                       X_name='gen1_0_4', Y_name='gen2_0_4',
                       kfds=5, method='eqdist', seed=7)

pms = comparison_tests(source='cnn_news', 
                       X=red_or, Y=red_gen2_0_4,
                       X_name='or', Y_name='gen2_0_4',
                       kfds=5, method='eqdist', seed=7)

crkfdsafsdvc = comparison_tests(source='cnn_news', 
                                X=red_or, Y=red_gen1_0_4,
                                X_name='or', Y_name='gen1_0_4',
                                kfds=5, method='eqdist', seed=7)

cxzvxvasvawdvdv = comparison_tests(source='cnn_news', 
                                   X=red_gen1_0_4, Y=red_gen1_1_0_4,
                                   X_name='gen1_0_4', Y_name='gen1_1_0_4',
                                   kfds=5, method='eqdist', seed=7)

dasvsvcsavac = comparison_tests(source='cnn_news', 
                                X=red_or, Y=red_gen1_1_0_4,
                                X_name='or', Y_name='gen1_1_0_4',
                                kfds=5, method='eqdist', seed=7)


## ball ################################
# quora
sdfqfqc = comparison_tests(source='quora', 
                           X=red_gen1_0_4, Y=red_gen2_0_4,
                           X_name='gen1_0_4', Y_name='gen2_0_4',
                           kfds=5, method='ball', seed=7)

tmtmdh = comparison_tests(source='quora', 
                          X=red_or, Y=red_gen2_0_4,
                          X_name='or', Y_name='gen2_0_4',
                          kfds=5, method='ball', seed=7)

ndfbf = comparison_tests(source='quora', 
                         X=red_or, Y=red_gen1_0_4,
                         X_name='or', Y_name='gen1_0_4',
                         kfds=5, method='ball', seed=7)

cvfwvfvw = comparison_tests(source='quora', 
                            X=red_gen1_0_4, Y=red_gen1_1_0_4,
                            X_name='gen1_0_4', Y_name='gen1_1_0_4',
                            kfds=5, method='ball', seed=7)

wfvvwfe = comparison_tests(source='quora', 
                           X=red_or, Y=red_gen1_1_0_4,
                           X_name='or', Y_name='gen1_1_0_4',
                           kfds=5, method='ball', seed=7)


# squad_2
cdaCxsc = comparison_tests(source='squad_2', 
                           X=red_gen1_0_4, Y=red_gen2_0_4,
                           X_name='gen1_0_4', Y_name='gen2_0_4',
                           kfds=5, method='ball', seed=7)

sxcasv = comparison_tests(source='squad_2', 
                          X=red_or, Y=red_gen2_0_4,
                          X_name='or', Y_name='gen2_0_4',
                          kfds=5, method='ball', seed=7)

scvascvbfb = comparison_tests(source='squad_2', 
                              X=red_or, Y=red_gen1_0_4,
                              X_name='or', Y_name='gen1_0_4',
                              kfds=5, method='ball', seed=7)

bgbgbggbbg = comparison_tests(source='squad_2', 
                              X=red_gen1_0_4, Y=red_gen1_1_0_4,
                              X_name='gen1_0_4', Y_name='gen1_1_0_4',
                              kfds=5, method='ball', seed=7)

asaas = comparison_tests(source='squad_2', 
                         X=red_or, Y=red_gen1_1_0_4,
                         X_name='or', Y_name='gen1_1_0_4',
                         kfds=5, method='ball', seed=7)


# cnn_news
cdcacdcsdeqfcq = comparison_tests(source='cnn_news', 
                                  X=red_gen1_0_4, Y=red_gen2_0_4,
                                  X_name='gen1_0_4', Y_name='gen2_0_4',
                                  kfds=5, method='ball', seed=7)

csacac3 = comparison_tests(source='cnn_news', 
                           X=red_or, Y=red_gen2_0_4,
                           X_name='or', Y_name='gen2_0_4',
                           kfds=5, method='ball', seed=7)

xcvxcvsvd4 = comparison_tests(source='cnn_news', 
                              X=red_or, Y=red_gen1_0_4,
                              X_name='or', Y_name='gen1_0_4',
                              kfds=5, method='ball', seed=7)

dsaf1 = comparison_tests(source='cnn_news', 
                         X=red_gen1_0_4, Y=red_gen1_1_0_4,
                         X_name='gen1_0_4', Y_name='gen1_1_0_4',
                         kfds=5, method='ball', seed=7)

abc13ds = comparison_tests(source='cnn_news', 
                           X=red_or, Y=red_gen1_1_0_4,
                           X_name='or', Y_name='gen1_1_0_4',
                           kfds=5, method='ball', seed=7)



'
temp 0.7
'

## temp 0.7 - eqdist ################################
# quora
k = comparison_tests(source='quora', 
                     X=red_gen1_0_7, Y=red_gen2_0_7,
                     X_name='gen1_0_7', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

l = comparison_tests(source='quora', 
                     X=red_or, Y=red_gen2_0_7,
                     X_name='or', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

m = comparison_tests(source='quora', 
                     X=red_or, Y=red_gen1_0_7,
                     X_name='or', Y_name='gen1_0_7',
                     kfds=5, method='eqdist', seed=7)

n = comparison_tests(source='quora', 
                     X=red_gen1_0_7, Y=red_gen1_1_0_7,
                     X_name='gen1_0_7', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)

o = comparison_tests(source='quora', 
                     X=red_or, Y=red_gen1_1_0_7,
                     X_name='or', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)


# squad_2
p = comparison_tests(source='squad_2', 
                     X=red_gen1_0_7, Y=red_gen2_0_7,
                     X_name='gen1_0_7', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

q = comparison_tests(source='squad_2', 
                     X=red_or, Y=red_gen2_0_7,
                     X_name='or', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

r = comparison_tests(source='squad_2', 
                     X=red_or, Y=red_gen1_0_7,
                     X_name='or', Y_name='gen1_0_7',
                     kfds=5, method='eqdist', seed=7)

s = comparison_tests(source='squad_2', 
                     X=red_gen1_0_7, Y=red_gen1_1_0_7,
                     X_name='gen1_0_7', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)

t = comparison_tests(source='squad_2', 
                     X=red_or, Y=red_gen1_1_0_7,
                     X_name='or', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)


# cnn_news
u = comparison_tests(source='cnn_news', 
                     X=red_gen1_0_7, Y=red_gen2_0_7,
                     X_name='gen1_0_7', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

v = comparison_tests(source='cnn_news', 
                     X=red_or, Y=red_gen2_0_7,
                     X_name='or', Y_name='gen2_0_7',
                     kfds=5, method='eqdist', seed=7)

w = comparison_tests(source='cnn_news', 
                     X=red_or, Y=red_gen1_0_7,
                     X_name='or', Y_name='gen1_0_7',
                     kfds=5, method='eqdist', seed=7)

x = comparison_tests(source='cnn_news', 
                     X=red_gen1_0_7, Y=red_gen1_1_0_7,
                     X_name='gen1_0_7', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)

y = comparison_tests(source='cnn_news', 
                     X=red_or, Y=red_gen1_1_0_7,
                     X_name='or', Y_name='gen1_1_0_7',
                     kfds=5, method='eqdist', seed=7)


## temp 0.7 - ball ################################
# quora
z = comparison_tests(source='quora', 
                     X=red_gen1_0_7, Y=red_gen2_0_7,
                     X_name='gen1_0_7', Y_name='gen2_0_7',
                     kfds=5, method='ball', seed=7)

aa = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_0_7,
                      X_name='or', Y_name='gen2_0_7',
                      kfds=5, method='ball', seed=7)

ab = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_0_7,
                      X_name='or', Y_name='gen1_0_7',
                      kfds=5, method='ball', seed=7)

ac = comparison_tests(source='quora', 
                      X=red_gen1_0_7, Y=red_gen1_1_0_7,
                      X_name='gen1_0_7', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)

ad = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_0_7,
                      X_name='or', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)


# squad_2
ae = comparison_tests(source='squad_2', 
                      X=red_gen1_0_7, Y=red_gen2_0_7,
                      X_name='gen1_0_7', Y_name='gen2_0_7',
                      kfds=5, method='ball', seed=7)

af = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_0_7,
                      X_name='or', Y_name='gen2_0_7',
                      kfds=5, method='ball', seed=7)

ag = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_0_7,
                      X_name='or', Y_name='gen1_0_7',
                      kfds=5, method='ball', seed=7)

ah = comparison_tests(source='squad_2', 
                      X=red_gen1_0_7, Y=red_gen1_1_0_7,
                      X_name='gen1_0_7', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)

ai = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_0_7,
                      X_name='or', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)


# cnn_news
aj = comparison_tests(source='cnn_news', 
                      X=red_gen1_0_7, Y=red_gen2_0_7,
                      X_name='gen1_0_7', Y_name='gen2_0_7',
                      kfds=5, method='ball', seed=7)

ak = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_0_7,
                      X_name='or', Y_name='gen2_0_7',
                      kfds=5, method='ball', seed=7)

al = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_0_7,
                      X_name='or', Y_name='gen1_0_7',
                      kfds=5, method='ball', seed=7)

am = comparison_tests(source='cnn_news', 
                      X=red_gen1_0_7, Y=red_gen1_1_0_7,
                      X_name='gen1_0_7', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)

an = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_0_7,
                      X_name='or', Y_name='gen1_1_0_7',
                      kfds=5, method='ball', seed=7)



'
temp 1.0
'

## eqdist ################################
ao = comparison_tests(source='quora', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

ap = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

aq = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='eqdist', seed=7)

ar = comparison_tests(source='quora', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)

as = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)


at = comparison_tests(source='squad_2', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

au = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

av = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='eqdist', seed=7)

aw = comparison_tests(source='squad_2', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)

ax = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)


ay = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

az = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='eqdist', seed=7)

ba = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='eqdist', seed=7)

bb = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)

bc = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='eqdist', seed=7)


## ball ################################
bs = comparison_tests(source='quora', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

bt = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

bu = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='ball', seed=7)

bv = comparison_tests(source='quora', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)

bw = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)


bx = comparison_tests(source='squad_2', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

by = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

bz = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='ball', seed=7)

ca = comparison_tests(source='squad_2', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)

cb = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)


cc = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_0, Y=red_gen2_1_0,
                      X_name='gen1_1_0', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

cd = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_1_0,
                      X_name='or', Y_name='gen2_1_0',
                      kfds=5, method='ball', seed=7)

ce = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_0,
                      X_name='or', Y_name='gen1_1_0',
                      kfds=5, method='ball', seed=7)

cf = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_0, Y=red_gen1_1_1_0,
                      X_name='gen1_1_0', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)

cg = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_1_0,
                      X_name='or', Y_name='gen1_1_1_0',
                      kfds=5, method='ball', seed=7)


'
temp 1.5
'

## eqdist ################################
bd = comparison_tests(source='quora', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

be = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

bf = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='eqdist', seed=7)

bg = comparison_tests(source='quora', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)

bh = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)


bi = comparison_tests(source='squad_2', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

bj = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

bk = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='eqdist', seed=7)

bl = comparison_tests(source='squad_2', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)

bm = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)


bn = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

bo = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='eqdist', seed=7)

bp = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='eqdist', seed=7)

bq = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)

br = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='eqdist', seed=7)


## ball ################################
ch = comparison_tests(source='quora', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

ci = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

cj = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='ball', seed=7)

ck = comparison_tests(source='quora', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)

cl = comparison_tests(source='quora', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)


cm = comparison_tests(source='squad_2', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

cn = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

co = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='ball', seed=7)

cp = comparison_tests(source='squad_2', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)

cq = comparison_tests(source='squad_2', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)


cr = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_5, Y=red_gen2_1_5,
                      X_name='gen1_1_5', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

cs = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen2_1_5,
                      X_name='or', Y_name='gen2_1_5',
                      kfds=5, method='ball', seed=7)

ct = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_5,
                      X_name='or', Y_name='gen1_1_5',
                      kfds=5, method='ball', seed=7)

cu = comparison_tests(source='cnn_news', 
                      X=red_gen1_1_5, Y=red_gen1_1_1_5,
                      X_name='gen1_1_5', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)

cv = comparison_tests(source='cnn_news', 
                      X=red_or, Y=red_gen1_1_1_5,
                      X_name='or', Y_name='gen1_1_1_5',
                      kfds=5, method='ball', seed=7)
