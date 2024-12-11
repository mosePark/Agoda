'
distance 독립성 검정
'

# MEASURING AND TESTING DEPENDENCE BY CORRELATION OF DISTANCES, the annals of stat, 2008

mapping = function(raw_dat1, raw_dat2, raw_dat3, repN=10, standardize=T, K=3){
  # standardize in a columnwise manner
  if (standardize){
    dat1 = scale(raw_dat1)
    dat2 = scale(raw_dat2)
    dat3 = scale(raw_dat3)
  }else{
    dat1 = raw_dat1
    dat2 = raw_dat2
    dat3 = raw_dat3
  }
  
  dat2_cl_list = list()
  dat3_cl_list = list()
  dat2_cl_totwithin = rep(NA, repN)
  dat3_cl_totwithin = rep(NA, repN)
  
  for (i in 1:repN){
    dat2_cl_list[[i]] = kmeans(dat2, K)
    dat3_cl_list[[i]] = kmeans(dat3, K)
    
    dat2_cl_totwithin[i] = dat2_cl_list[[i]]$tot.withinss
    dat3_cl_totwithin[i] = dat3_cl_list[[i]]$tot.withinss
  }
  dat2_cl = dat2_cl_list[[which.min(dat2_cl_totwithin)]]
  dat3_cl = dat3_cl_list[[which.min(dat3_cl_totwithin)]]
  
  dat2_on_dat1_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  dat3_on_dat1_cent = matrix(NA, nrow=K, ncol=ncol(dat1))
  
  for (k in 1:K){
    dat2_on_dat1_cent[k,] = apply(dat1[dat2_cl$cluster==k, ],2,mean)
    dat3_on_dat1_cent[k,] = apply(dat1[dat3_cl$cluster==k, ],2,mean)
  }
  
  dat2_on_dat1_dist = apply((dat1 -dat2_on_dat1_cent[dat2_cl$cluster,])^2, 1, sum)
  dat3_on_dat1_dist = apply((dat1 -dat3_on_dat1_cent[dat3_cl$cluster,])^2, 1, sum)

  return(list(d2ond1 = dat2_on_dat1_dist,
              d3ond1 = dat3_on_dat1_dist
              )
         )
}

library(energy)

? dcov.test

save_dir <- "D:/ebd_matrix"

# EBD
or_emb <- readRDS(file.path(save_dir, "or_emb.rds"))

gen1_emb_0_1 <- readRDS(file.path(save_dir, "gen1_emb_0_1.rds"))
gen2_emb_0_1 <- readRDS(file.path(save_dir, "gen2_emb_0_1.rds"))
gen1_1_emb_0_1 <- readRDS(file.path(save_dir, "gen1_1_emb_0_1.rds"))

gen1_emb_0_7 <- readRDS(file.path(save_dir, "gen1_emb_0_7.rds"))
gen2_emb_0_7 <- readRDS(file.path(save_dir, "gen2_emb_0_7.rds"))
gen1_1_emb_0_7 <- readRDS(file.path(save_dir, "gen1_1_emb_0_7.rds"))

gen1_emb_1_5 <- readRDS(file.path(save_dir, "gen1_emb_1_5.rds"))
gen2_emb_1_5 <- readRDS(file.path(save_dir, "gen2_emb_1_5.rds"))
gen1_1_emb_1_5 <- readRDS(file.path(save_dir, "gen1_1_emb_1_5.rds"))

# SVD
SVD_or <- readRDS(file.path(save_dir, "SVD_or.rds"))

SVD_gen1_0_1 <- readRDS(file.path(save_dir, "SVD_gen1_0_1.rds"))
SVD_gen2_0_1 <- readRDS(file.path(save_dir, "SVD_gen2_0_1.rds"))
SVD_gen1_1_0_1 <- readRDS(file.path(save_dir, "SVD_gen1_1_0_1.rds"))

SVD_gen1_0_7 <- readRDS(file.path(save_dir, "SVD_gen1_0_7.rds"))
SVD_gen2_0_7 <- readRDS(file.path(save_dir, "SVD_gen2_0_7.rds"))
SVD_gen1_1_0_7 <- readRDS(file.path(save_dir, "SVD_gen1_1_0_7.rds"))

SVD_gen1_1_5 <- readRDS(file.path(save_dir, "SVD_gen1_1_5.rds"))
SVD_gen2_1_5 <- readRDS(file.path(save_dir, "SVD_gen2_1_5.rds"))
SVD_gen1_1_1_5 <- readRDS(file.path(save_dir, "SVD_gen1_1_1_5.rds"))

dim = 2

# Reduction
red_or = or_emb%*%SVD_or$v[,1:dim]

red_gen1_0_1 = gen1_emb_0_1%*%SVD_gen1_0_1$v[,1:dim]
red_gen2_0_1 = gen2_emb_0_1%*%SVD_gen2_0_1$v[,1:dim]
red_gen1_1_0_1 = gen1_1_emb_0_1%*%SVD_gen1_1_0_1$v[,1:dim]

red_gen1_0_7 = gen1_emb_0_7%*%SVD_gen1_0_7$v[,1:dim]
red_gen2_0_7 = gen2_emb_0_7%*%SVD_gen2_0_7$v[,1:dim]
red_gen1_1_0_7 = gen1_1_emb_0_7%*%SVD_gen1_1_0_7$v[,1:dim]

red_gen1_1_5 = gen1_emb_1_5%*%SVD_gen1_1_5$v[,1:dim]
red_gen2_1_5 = gen2_emb_1_5%*%SVD_gen2_1_5$v[,1:dim]
red_gen1_1_1_5 = gen1_1_emb_1_5%*%SVD_gen1_1_1_5$v[,1:dim]

# Independence testing

org1g2_07 = mapping(red_or, red_gen1_0_7, red_gen2_0_7)

dcov.test(org1g2_07$d2ond1, org1g2_07$d3ond1)
##########
### 에러 : 에러: 크기가 7.7 Gb인 벡터를 할당할 수 없습니다 (3만 * 3만 거리행렬을 메모리가 할당하기 힘듦)
#########


sample_size <- 3000

B = 10

p_values <- numeric(B)

for (i in 1:B) {

  sample_indices <- sample(1:nrow(red_or), size = sample_size, replace = FALSE)
  red_or_sample <- red_or[sample_indices, ]
  red_gen1_0_7_sample <- red_gen1_0_7[sample_indices, ]
  red_gen2_0_7_sample <- red_gen2_0_7[sample_indices, ]
  

  org1g2_07 <- mapping(red_or_sample, red_gen1_0_7_sample, red_gen2_0_7_sample)
  p_values[i] <- dcov.test(org1g2_07$d2ond1, org1g2_07$d3ond1, R = 100)$p.value
}

cat("p-values from bootstrapping:\n", p_values, "\n")
cat("Mean p-value:", mean(p_values), "\n")
cat("Proportion of significant results (p < 0.05):", mean(p_values < 0.05), "\n")

#

org1g2_07 <- mapping(red_or, red_gen1_0_7, red_gen2_0_7)

d2ond1 = org1g2_07$d2ond1
d3ond1 = org1g2_07$d3ond1

# 피어슨 상관분석
correlation_pearson <- cor(d2ond1, d3ond1, method = "pearson")
print(paste("피어슨 상관계수:", correlation_pearson))

# 스피어만 상관분석 (데이터가 정규성을 만족하지 않을 경우)
correlation_spearman <- cor(d2ond1, d3ond1, method = "spearman")
print(paste("스피어만 상관계수:", correlation_spearman))

# 상관분석 결과를 확인하는 테스트
test_pearson <- cor.test(d2ond1, d3ond1, method = "pearson")
print("피어슨 상관분석 테스트 결과:")
print(test_pearson)

test_spearman <- cor.test(d2ond1, d3ond1, method = "spearman")
print("스피어만 상관분석 테스트 결과:")
print(test_spearman)

# chi sq

# 교차표 생성
contingency_table <- table(d2ond1, d3ond1)

# Chi-Square 독립성 검정
chisq_result <- chisq.test(contingency_table)

# 결과 출력
print(chisq_result)

########################
# 에러: 크기가 7.5 Gb인 벡터를 할당할 수 없습니다
########################

# 범주를 줄이기 위해 binning 수행
d2ond1_binned <- cut(d2ond1, breaks = 1000, labels = FALSE)  # ~개의 구간으로 나누기
d3ond1_binned <- cut(d3ond1, breaks = 1000, labels = FALSE)

# 새로운 교차표 생성 및 검정
contingency_table <- table(d2ond1_binned, d3ond1_binned)
chisq_result <- chisq.test(contingency_table)

# 결과 출력
print(chisq_result)
## 의미 있는 코드인지?



set.seed(123)
sample_size <- 10000  # 적절한 크기의 샘플링
indices <- sample(seq_along(d2ond1), size = sample_size)

d2ond1_sample <- d2ond1[indices]
d3ond1_sample <- d3ond1[indices]

# 교차표 생성 및 검정
contingency_table <- table(d2ond1_sample, d3ond1_sample)
chisq_result <- chisq.test(contingency_table)

# 결과 출력
print(chisq_result)


# 범주의 수가 많거나 데이터가 연속형인 경우, Chi-Square 검정 대신 비모수적 검정을 사용하는 것이 적합
library(infotheo)

# 상호정보량 계산
mutual_info <- mutinformation(d2ond1, d3ond1)

# 결과 출력
print(mutual_info)


