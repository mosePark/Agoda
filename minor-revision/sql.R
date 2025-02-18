setwd("C:/Users/mose/Desktop/result_minor_revision")


rdata_files <- list.files(pattern = "\\.rdata$") ; rdata_files

# agoda 파일들
load("agoda_0_1.rdata")
ago_0_1 <- result_list

load("agoda_0_4.rdata")
ago_0_4 <- result_list

load("agoda_0_7.rdata")
ago_0_7 <- result_list

load("agoda_1_0.rdata")
ago_1_0 <- result_list

load("agoda_1_5.rdata")
ago_1_5 <- result_list

# cnn_news 파일들
load("cnn_news_0_1.rdata")
cnn_news_0_1 <- result_list

load("cnn_news_0_4.rdata")
cnn_news_0_4 <- result_list

load("cnn_news_0_7.rdata")
cnn_news_0_7 <- result_list

load("cnn_news_1_0.rdata")
cnn_news_1_0 <- result_list

load("cnn_news_1_5.rdata")
cnn_news_1_5 <- result_list

# quora 파일들
load("quora_0_1.rdata")
quora_0_1 <- result_list

load("quora_0_4.rdata")
quora_0_4 <- result_list

load("quora_0_7.rdata")
quora_0_7 <- result_list

load("quora_1_0.rdata")
quora_1_0 <- result_list

load("quora_1_5.rdata")
quora_1_5 <- result_list

# squad 파일들
load("squad_2_0_1.rdata")
squad_2_0_1 <- result_list

load("squad_2_0_4.rdata")
squad_2_0_4 <- result_list

load("squad_2_0_7.rdata")
squad_2_0_7 <- result_list

load("squad_2_1_0.rdata")
squad_2_1_0 <- result_list

load("squad_2_1_5.rdata")
squad_2_1_5 <- result_list

extract <- function(result, dim) {
  res <- result[[dim]]
  list(
    gen1_gen2   = res[rownames(res) %in% rows, "gen1_gen2"],
    or_gen2     = res[rownames(res) %in% rows, "or_gen2"],
    gen1_gen1_1 = res[rownames(res) %in% rows, "gen1_gen1_1"],
    or_gen1_1   = res[rownames(res) %in% rows, "or_gen1_1"]
  )
}
rows = c("t.test_2",  "t.test_3",  "t.test_4", "t.test_5")
#########################
# dim = 2 table 채우기 [[1]]
#########################

# --- agoda ---
ago_0_1_dim2 <- extract(ago_0_1, 1)
ago_0_4_dim2 <- extract(ago_0_4, 1)
ago_0_7_dim2 <- extract(ago_0_7, 1)
ago_1_0_dim2 <- extract(ago_1_0, 1)
ago_1_5_dim2 <- extract(ago_1_5, 1)

# --- cnn_news ---
cnn_news_0_1_dim2 <- extract(cnn_news_0_1, 1)
cnn_news_0_4_dim2 <- extract(cnn_news_0_4, 1)
cnn_news_0_7_dim2 <- extract(cnn_news_0_7, 1)
cnn_news_1_0_dim2 <- extract(cnn_news_1_0, 1)
cnn_news_1_5_dim2 <- extract(cnn_news_1_5, 1)

# --- quora ---
quora_0_1_dim2 <- extract(quora_0_1, 1)
quora_0_4_dim2 <- extract(quora_0_4, 1)
quora_0_7_dim2 <- extract(quora_0_7, 1)
quora_1_0_dim2 <- extract(quora_1_0, 1)
quora_1_5_dim2 <- extract(quora_1_5, 1)

# --- squad ---
squad_2_0_1_dim2 <- extract(squad_2_0_1, 1)
squad_2_0_4_dim2 <- extract(squad_2_0_4, 1)
squad_2_0_7_dim2 <- extract(squad_2_0_7, 1)
squad_2_1_0_dim2 <- extract(squad_2_1_0, 1)
squad_2_1_5_dim2 <- extract(squad_2_1_5, 1)

#########################
# dim = 10 table 채우기 [[5]]
#########################

# --- agoda ---
ago_0_1_dim10 <- extract(ago_0_1, 5)
ago_0_4_dim10 <- extract(ago_0_4, 5)
ago_0_7_dim10 <- extract(ago_0_7, 5)
ago_1_0_dim10 <- extract(ago_1_0, 5)
ago_1_5_dim10 <- extract(ago_1_5, 5)

# --- cnn_news ---
cnn_news_0_1_dim10 <- extract(cnn_news_0_1, 5)
cnn_news_0_4_dim10 <- extract(cnn_news_0_4, 5)
cnn_news_0_7_dim10 <- extract(cnn_news_0_7, 5)
cnn_news_1_0_dim10 <- extract(cnn_news_1_0, 5)
cnn_news_1_5_dim10 <- extract(cnn_news_1_5, 5)

# --- quora ---
quora_0_1_dim10 <- extract(quora_0_1, 5)
quora_0_4_dim10 <- extract(quora_0_4, 5)
quora_0_7_dim10 <- extract(quora_0_7, 5)
quora_1_0_dim10 <- extract(quora_1_0, 5)
quora_1_5_dim10 <- extract(quora_1_5, 5)

# --- squad ---
squad_2_0_1_dim10 <- extract(squad_2_0_1, 5)
squad_2_0_4_dim10 <- extract(squad_2_0_4, 5)
squad_2_0_7_dim10 <- extract(squad_2_0_7, 5)
squad_2_1_0_dim10 <- extract(squad_2_1_0, 5)
squad_2_1_5_dim10 <- extract(squad_2_1_5, 5)

#########################
# dim = 20 table 채우기 [[7]]
#########################

# --- agoda ---
ago_0_1_dim20 <- extract(ago_0_1, 7)
ago_0_4_dim20 <- extract(ago_0_4, 7)
ago_0_7_dim20 <- extract(ago_0_7, 7)
ago_1_0_dim20 <- extract(ago_1_0, 7)
ago_1_5_dim20 <- extract(ago_1_5, 7)

# --- cnn_news ---
cnn_news_0_1_dim20 <- extract(cnn_news_0_1, 7)
cnn_news_0_4_dim20 <- extract(cnn_news_0_4, 7)
cnn_news_0_7_dim20 <- extract(cnn_news_0_7, 7)
cnn_news_1_0_dim20 <- extract(cnn_news_1_0, 7)
cnn_news_1_5_dim20 <- extract(cnn_news_1_5, 7)

# --- quora ---
quora_0_1_dim20 <- extract(quora_0_1, 7)
quora_0_4_dim20 <- extract(quora_0_4, 7)
quora_0_7_dim20 <- extract(quora_0_7, 7)
quora_1_0_dim20 <- extract(quora_1_0, 7)
quora_1_5_dim20 <- extract(quora_1_5, 7)

# --- squad ---
squad_2_0_1_dim20 <- extract(squad_2_0_1, 7)
squad_2_0_4_dim20 <- extract(squad_2_0_4, 7)
squad_2_0_7_dim20 <- extract(squad_2_0_7, 7)
squad_2_1_0_dim20 <- extract(squad_2_1_0, 7)
squad_2_1_5_dim20 <- extract(squad_2_1_5, 7)

################################################################################
ago_0_1_dim2$or_gen1_1
cnn_news_0_1_dim2$or_gen1_1
quora_0_1_dim2$or_gen1_1
squad_2_0_1_dim2$or_gen1_1

# rho = 0_1
mat_0_1 <- rbind(
  Review = ago_0_1_dim2$or_gen1_1,
  CNN    = cnn_news_0_1_dim2$or_gen1_1,
  Quora  = quora_0_1_dim2$or_gen1_1,
  SQuAD2 = squad_2_0_1_dim2$or_gen1_1
)
colnames(mat_0_1) <- c("K=2", "K=3", "K=4", "K=5")
print(mat_0_1)

# rho = 0_4
mat_0_4 <- rbind(
  Review = ago_0_4_dim2$or_gen1_1,
  CNN    = cnn_news_0_4_dim2$or_gen1_1,
  Quora  = quora_0_4_dim2$or_gen1_1,
  SQuAD2 = squad_2_0_4_dim2$or_gen1_1
)
colnames(mat_0_4) <- c("K=2", "K=3", "K=4", "K=5")
print(mat_0_4)

# rho = 0_7
mat_0_7 <- rbind(
  Review = ago_0_7_dim2$or_gen1_1,
  CNN    = cnn_news_0_7_dim2$or_gen1_1,
  Quora  = quora_0_7_dim2$or_gen1_1,
  SQuAD2 = squad_2_0_7_dim2$or_gen1_1
)
colnames(mat_0_7) <- c("K=2", "K=3", "K=4", "K=5")
print(mat_0_7)

# rho = 1_0
mat_1_0 <- rbind(
  Review = ago_1_0_dim2$or_gen1_1,
  CNN    = cnn_news_1_0_dim2$or_gen1_1,
  Quora  = quora_1_0_dim2$or_gen1_1,
  SQuAD2 = squad_2_1_0_dim2$or_gen1_1
)
colnames(mat_1_0) <- c("K=2", "K=3", "K=4", "K=5")
print(mat_1_0)

# rho = 1_5
mat_1_5 <- rbind(
  Review = ago_1_5_dim2$or_gen1_1,
  CNN    = cnn_news_1_5_dim2$or_gen1_1,
  Quora  = quora_1_5_dim2$or_gen1_1,
  SQuAD2 = squad_2_1_5_dim2$or_gen1_1
)
colnames(mat_1_5) <- c("K=2", "K=3", "K=4", "K=5")
print(mat_1_5)

final_mat <- rbind(mat_0_1, mat_0_4, mat_0_7, mat_1_0, mat_1_5)
print(final_mat)



squad_2_1_0_dim2$gen1_gen2
squad_2_1_0_dim2$or_gen2
squad_2_1_0_dim2$gen1_gen1_1
squad_2_1_0_dim2$



# rho 그룹들
rho_values <- c("0_1", "0_4", "0_7", "1_0", "1_5")

# 예시: dim_label과 var_name 설정 (예: "dim2"와 "or_gen1_1")
dim_label <- "dim20"          # 예: "dim2", "dim10", "dim20" 등
var_name  <- "gen1_gen2"     # 예: "or_gen1_1", "or_gen2", "gen1_gen1_1", "gen1_gen2", " 등

# 각 rho 값별로 해당 변수 값을 추출하여 행렬을 생성하는 함수
create_mat <- function(rho, dim_label, var_name) {
  # 각 데이터셋의 객체명 동적 생성  
  # agoda, cnn_news, quora는 "ago_", "cnn_news_", "quora_"로,
  # squad는 "squad_2_" 형식임에 유의합니다.
  review_obj <- get(paste0("ago_", rho, "_", dim_label))
  cnn_obj    <- get(paste0("cnn_news_", rho, "_", dim_label))
  quora_obj  <- get(paste0("quora_", rho, "_", dim_label))
  squad_obj  <- get(paste0("squad_2_", rho, "_", dim_label))
  
  mat <- rbind(
    Review = review_obj[[ var_name ]],
    CNN    = cnn_obj[[ var_name ]],
    Quora  = quora_obj[[ var_name ]],
    SQuAD2 = squad_obj[[ var_name ]]
  )
  colnames(mat) <- c("K=2", "K=3", "K=4", "K=5")
  return(mat)
}

# 각 rho 값에 대해 행렬 생성하여 리스트에 저장
mat_list <- lapply(rho_values, create_mat, dim_label = dim_label, var_name = var_name)
names(mat_list) <- rho_values

# 최종 행렬: 리스트에 저장된 행렬들을 행방향으로 결합
final_mat <- do.call(rbind, mat_list)
print(final_mat)
