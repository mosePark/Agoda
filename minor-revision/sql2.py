'
추가 테이블 만들기, 차원에 따라 값 다 찾아서 넣기
'

setwd("C:/Users/mose/Desktop/results_comp")

rdata_files <- list.files(pattern = "\\.[Rr]data$", ignore.case = TRUE)

load("agora_0_1.RData")
ago_0_1 <- result_list

load("agora_0_4.RData")
ago_0_4 <- result_list

load("agora_0_7.RData")
ago_0_7 <- result_list

load("agora_1_0.RData")
ago_1_0 <- result_list

load("agora_1_5.RData")
ago_1_5 <- result_list

load("cnn_news_0_1.RData")
cnn_news_0_1 <- result_list

load("cnn_news_0_4.RData")
cnn_news_0_4 <- result_list

load("cnn_news_0_7.RData")
cnn_news_0_7 <- result_list

load("cnn_news_1_0.RData")
cnn_news_1_0 <- result_list

load("cnn_news_1_5.RData")
cnn_news_1_5 <- result_list

load("quora_0_1.RData")
quora_0_1 <- result_list

load("quora_0_4.RData")
quora_0_4 <- result_list

load("quora_0_7.RData")
quora_0_7 <- result_list

load("quora_1_0.RData")
quora_1_0 <- result_list

load("quora_1_5.RData")
quora_1_5 <- result_list

load("squad_2_0_1.RData")
squad_2_0_1 <- result_list

load("squad_2_0_4.RData")
squad_2_0_4 <- result_list

load("squad_2_0_7.RData")
squad_2_0_7 <- result_list

load("squad_2_1_0.RData")
squad_2_1_0 <- result_list

load("squad_2_1_5.RData")
squad_2_1_5 <- result_list

ago_0_1$hotelling
ago_0_1$np

create_result_table <- function(red_dim = 2, col_name = "or_gen1_1") {
  # 관심있는 rho 값들 (파일명에 포함된 부분)
  rho_values <- c("0_1", "0_4", "0_7", "1_0", "1_5")
  
  # 각 데이터셋에 대한 객체 이름 생성 함수
  datasets <- list(
    Review = function(rho) get(paste0("ago_", rho)),
    CNN    = function(rho) get(paste0("cnn_news_", rho)),
    Quora  = function(rho) get(paste0("quora_", rho)),
    SQUAD2 = function(rho) get(paste0("squad_2_", rho))
  )
  
  # 결과를 저장할 빈 데이터프레임 (열: hotelling, np)
  res <- data.frame(hotelling = numeric(), np = numeric(), stringsAsFactors = FALSE)
  row_names <- c()
  
  # 내부 함수: 지정한 객체에서 red_dim 행과 선택한 칼럼의 값을 추출
  get_val <- function(obj, red_dim, col_name) {
    # 테이블의 행 이름은 문자형("2", "10", "20" 등)로 되어 있다고 가정
    val_hot <- obj$hotelling[as.character(red_dim), col_name]
    val_np  <- obj$np[as.character(red_dim), col_name]
    return(c(hotelling = val_hot, np = val_np))
  }
  
  # 각 rho 값과 데이터셋에 대해 값 추출
  for(rho in rho_values) {
    for(ds in names(datasets)) {
      obj <- datasets[[ds]](rho)
      vals <- get_val(obj, red_dim, col_name)
      res <- rbind(res, vals)
      row_names <- c(row_names, paste(rho, ds))
    }
  }
  
  rownames(res) <- row_names
  colnames(res) <- c('hotelling', 'np')
  return(res)
}


# "or_gen1_1", "or_gen2", "gen1_gen1_1", "gen1_gen2"

dim = 20
colnms = "gen1_gen1_1"

result_table <- create_result_table(red_dim = dim, col_name = colnms)
print(result_table)
