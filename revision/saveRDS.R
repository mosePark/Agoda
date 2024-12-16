rm(list = ls(all=TRUE))

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

set.seed(123)

'
Load Data
'

# "C:/UBAI/hug-ebd/0.7-hug-gen1-ebd.csv"

df <- read.csv("C:/UBAI/hug-ebd/ori-hug-ebd.csv", header=T)

df_1 <- read.csv("C:/UBAI/hug-ebd/0.1-hug-gen1-ebd.csv", header=T)
df_2 <- read.csv("C:/UBAI/hug-ebd/0.1-hug-gen2-ebd.csv", header=T)
df_3 <- read.csv("C:/UBAI/hug-ebd/0.1-hug-gen1-1-ebd.csv", header=T)

df_4 <- read.csv("C:/UBAI/hug-ebd/0.4-hug-gen1-ebd.csv", header=T)
df_5 <- read.csv("C:/UBAI/hug-ebd/0.4-hug-gen2-ebd.csv", header=T)
df_6 <- read.csv("C:/UBAI/hug-ebd/0.4-hug-gen1-1-ebd.csv", header=T)

df_7 <- read.csv("C:/UBAI/hug-ebd/0.7-hug-gen1-ebd.csv", header=T)
df_8 <- read.csv("C:/UBAI/hug-ebd/0.7-hug-gen2-ebd.csv", header=T)
df_9 <- read.csv("C:/UBAI/hug-ebd/0.7-hug-gen1-1-ebd.csv", header=T)

df_10 <- read.csv("C:/UBAI/hug-ebd/1.0-hug-gen1-ebd.csv", header=T)
df_11 <- read.csv("C:/UBAI/hug-ebd/1.0-hug-gen2-ebd.csv", header=T)
df_12 <- read.csv("C:/UBAI/hug-ebd/1.0-hug-gen1-1-ebd.csv", header=T)

df_13 <- read.csv("C:/UBAI/hug-ebd/1.5-hug-gen1-ebd.csv", header=T)
df_14 <- read.csv("C:/UBAI/hug-ebd/1.5-hug-gen2-ebd.csv", header=T)

# ori
ori_emb = make_matrix(df$ori.ebd)
SVD_ori = svd(ori_emb)

# 0.1
gen1_emb_0_1 = make_matrix(df_1$X0.1.gen1.ebd)
gen2_emb_0_1 = make_matrix(df_2$X0.1.gen2.ebd)
gen1_1_emb_0_1 = make_matrix(df_3$X0.1.gen1.1.ebd)

SVD_gen1_0_1 = svd(gen1_emb_0_1)
SVD_gen2_0_1 = svd(gen2_emb_0_1)
SVD_gen1_1_0_1 = svd(gen1_1_emb_0_1)

# 0.4
gen1_emb_0_4 = make_matrix(df_4$X0.4.gen1.ebd)
gen2_emb_0_4 = make_matrix(df_5$X0.4.gen2.ebd)
gen1_1_emb_0_4 = make_matrix(df_6$X0.4.gen1.1.ebd)

SVD_gen1_0_4 = svd(gen1_emb_0_4)
SVD_gen2_0_4 = svd(gen2_emb_0_4)
SVD_gen1_1_0_4 = svd(gen1_1_emb_0_4)

# 0.7
gen1_emb_0_7 = make_matrix(df_7$X0.7.gen1.ebd)
gen2_emb_0_7 = make_matrix(df_8$X0.7.gen2.ebd)
gen1_1_emb_0_7 = make_matrix(df_9$X0.7.gen1.1.ebd)

SVD_gen1_0_7 = svd(gen1_emb_0_7)
SVD_gen2_0_7 = svd(gen2_emb_0_7)
SVD_gen1_1_0_7 = svd(gen1_1_emb_0_7)

# 1.0
gen1_emb_1_0 = make_matrix(df_10$X1.0.gen1.ebd)
gen2_emb_1_0 = make_matrix(df_11$X1.0.gen2.ebd)
gen1_1_emb_1_0 = make_matrix(df_12$X1.0.gen1.1.ebd)

SVD_gen1_1_0 = svd(gen1_emb_1_0)
SVD_gen2_1_0 = svd(gen2_emb_1_0)
SVD_gen1_1_1_0 = svd(gen1_1_emb_1_0)

# 1.5
gen1_emb_1_5 = make_matrix(df_13$X1.5.gen1.ebd)
gen2_emb_1_5 = make_matrix(df_14$X1.5.gen2.ebd)

SVD_gen1_1_5 = svd(gen1_emb_1_5)
SVD_gen2_1_5 = svd(gen2_emb_1_5)




# save

saveRDS(SVD_ori, file = "C:/UBAI/hug-ebd/SVD_ori.rds")

saveRDS(SVD_gen1_0_1, file = "C:/UBAI/hug-ebd/SVD_gen1_0_1.rds")
saveRDS(SVD_gen2_0_1, file = "C:/UBAI/hug-ebd/SVD_gen2_0_1.rds")
saveRDS(SVD_gen1_1_0_1, file = "C:/UBAI/hug-ebd/SVD_gen1_1_0_1.rds")

saveRDS(SVD_gen1_0_4, file = "C:/UBAI/hug-ebd/SVD_gen1_0_4.rds")
saveRDS(SVD_gen2_0_4, file = "C:/UBAI/hug-ebd/SVD_gen2_0_4.rds")
saveRDS(SVD_gen1_1_0_4, file = "C:/UBAI/hug-ebd/SVD_gen1_1_0_4.rds")

saveRDS(SVD_gen1_0_7, file = "C:/UBAI/hug-ebd/SVD_gen1_0_7.rds")
saveRDS(SVD_gen2_0_7, file = "C:/UBAI/hug-ebd/SVD_gen2_0_7.rds")
saveRDS(SVD_gen1_1_0_7, file = "C:/UBAI/hug-ebd/SVD_gen1_1_0_7.rds")

saveRDS(SVD_gen1_1_0, file = "C:/UBAI/hug-ebd/SVD_gen1_1_0.rds")
saveRDS(SVD_gen2_1_0, file = "C:/UBAI/hug-ebd/SVD_gen2_1_0.rds")
saveRDS(SVD_gen1_1_1_0, file = "C:/UBAI/hug-ebd/SVD_gen1_1_1_0.rds")

# Loading SVD data
SVD_ori <- readRDS("C:/UBAI/hug-ebd/SVD_ori.rds")

SVD_gen1_0_1 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_0_1.rds")
SVD_gen2_0_1 <- readRDS("C:/UBAI/hug-ebd/SVD_gen2_0_1.rds")
SVD_gen1_1_0_1 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_1_0_1.rds")

SVD_gen1_0_4 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_0_4.rds")
SVD_gen2_0_4 <- readRDS("C:/UBAI/hug-ebd/SVD_gen2_0_4.rds")
SVD_gen1_1_0_4 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_1_0_4.rds")

SVD_gen1_0_7 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_0_7.rds")
SVD_gen2_0_7 <- readRDS("C:/UBAI/hug-ebd/SVD_gen2_0_7.rds")
SVD_gen1_1_0_7 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_1_0_7.rds")

SVD_gen1_1_0 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_1_0.rds")
SVD_gen2_1_0 <- readRDS("C:/UBAI/hug-ebd/SVD_gen2_1_0.rds")
SVD_gen1_1_1_0 <- readRDS("C:/UBAI/hug-ebd/SVD_gen1_1_1_0.rds")
