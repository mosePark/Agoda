library(ggplot2)

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

set.seed(241003)

setwd("E:/mose/data/ablation2")
# setwd("E:/mose/data/ablation2")

getwd()

df_4 <- read.csv("df_4.csv", header=T)
df_5 <- read.csv("df_5.csv", header=T)
df_6 <- read.csv("df_6.csv", header=T)


sz = 5000

sample_ind = sample(1:nrow(df_4), size=sz)

or_emb = make_matrix(df_6$ori.ebd[sample_ind])
gen1_emb = make_matrix(df_4$X0.7.gen1.ebd[sample_ind])
gen2_emb = make_matrix(df_5$X0.7.gen2.ebd[sample_ind])
gen1_1_emb = make_matrix(df_6$X0.7.gen1.1.ebd[sample_ind])



# 각 SVD 객체에서 특잇값 가져오기
singular_values_or <- SVD_or$d
singular_values_gen1 <- SVD_gen1$d
singular_values_gen2 <- SVD_gen2$d
singular_values_gen1_1 <- SVD_gen1_1$d

# 각 SVD의 분산 설명 비율 계산
explained_variance_or <- (singular_values_or^2) / sum(singular_values_or^2) * 100
explained_variance_gen1 <- (singular_values_gen1^2) / sum(singular_values_gen1^2) * 100
explained_variance_gen2 <- (singular_values_gen2^2) / sum(singular_values_gen2^2) * 100
explained_variance_gen1_1 <- (singular_values_gen1_1^2) / sum(singular_values_gen1_1^2) * 100

# Scree Plot 데이터 생성
scree_data_or <- data.frame(Dimension = seq_along(explained_variance_or), Variance = explained_variance_or, Type = "Original Embedding")
scree_data_gen1 <- data.frame(Dimension = seq_along(explained_variance_gen1), Variance = explained_variance_gen1, Type = "Generated Embedding 1")
scree_data_gen2 <- data.frame(Dimension = seq_along(explained_variance_gen2), Variance = explained_variance_gen2, Type = "Generated Embedding 2")
scree_data_gen1_1 <- data.frame(Dimension = seq_along(explained_variance_gen1_1), Variance = explained_variance_gen1_1, Type = "Generated Embedding 1.1")

# 데이터 결합
scree_data <- rbind(scree_data_or, scree_data_gen1, scree_data_gen2, scree_data_gen1_1)

# Scree Plot 그리기
ggplot(scree_data, aes(x = Dimension, y = Variance, color = Type)) +
  geom_bar(stat = "identity", position = "dodge", fill = "skyblue", alpha = 0.7) +
  geom_line() +
  geom_point() +
  geom_vline(xintercept = 15, linetype = "dashed", color = "red", size = 0.8) + # 15번째 주성분에 수직선 추가
  labs(title = "Scree Plot of Singular Values for Different Embeddings",
       x = "Principal Component",
       y = "Explained Variance (%)") +
  theme_minimal() +
  theme(legend.position = "top")


# 데이터프레임으로 결합
variance_table <- data.frame(
  Dimension = seq_along(explained_variance_or),
  Original_Embedding = explained_variance_or,
  Generated_Embedding_1 = explained_variance_gen1,
  Generated_Embedding_2 = explained_variance_gen2,
  Generated_Embedding_1_1 = explained_variance_gen1_1
)

# 테이블 출력
print(variance_table)

# 각 SVD의 상위 15차원까지 분산 설명 비율 합계 계산
cumulative_variance_or <- sum(explained_variance_or[1:2])
cumulative_variance_gen1 <- sum(explained_variance_gen1[1:2])
cumulative_variance_gen2 <- sum(explained_variance_gen2[1:2])
cumulative_variance_gen1_1 <- sum(explained_variance_gen1_1[1:2])

print(cumulative_variance_or) # dim15  :  54.6%
print(cumulative_variance_gen1) # dim15  :  61.4%
print(cumulative_variance_gen2) # dim15  :  61.5%
print(cumulative_variance_gen1_1) # dim15  :   64%
