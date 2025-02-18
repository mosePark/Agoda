rm(list = ls())

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

# cnn_news 파일들 , "std_result_ls"  "nstd_result_ls"
load("cnn_news_0_1.rdata")
cnn_news_0_1 <- nstd_result_ls

load("cnn_news_0_4.rdata")
print(result_list[[1]])
cnn_news_0_4 <- nstd_result_ls

load("cnn_news_0_7.rdata")
print(result_list[[1]])
cnn_news_0_7 <- nstd_result_ls

load("cnn_news_1_0.rdata")
cnn_news_1_0 <- nstd_result_ls

load("cnn_news_1_5.rdata")
cnn_news_1_5 <- nstd_result_ls

# quora 파일들
load("quora_0_1.rdata")
quora_0_1 <- nstd_result_ls

load("quora_0_4.rdata")
quora_0_4 <- nstd_result_ls

load("quora_0_7.rdata")
quora_0_7 <- nstd_result_ls

load("quora_1_0.rdata")
quora_1_0 <- nstd_result_ls

load("quora_1_5.rdata")
quora_1_5 <- nstd_result_ls

# squad 파일들
load("squad_2_0_1.rdata")
squad_2_0_1 <- nstd_result_ls

load("squad_2_0_4.rdata")
squad_2_0_4 <- nstd_result_ls

load("squad_2_0_7.rdata")
squad_2_0_7 <- nstd_result_ls

load("squad_2_1_0.rdata")
squad_2_1_0 <- nstd_result_ls

load("squad_2_1_5.rdata")
squad_2_1_5 <- nstd_result_ls

# red_dim 값 (각 리스트의 순서에 해당)
red_dims <- c(2, 3, 4, 5, 10, 15, 20)
# 비교할 테스트 이름들 (행 이름)
tests <- c("t.test_2", "t.test_3", "t.test_4", "t.test_5")
# 비교할 유형 (열 이름)
columns <- c("or_gen1_1", "or_gen2", "gen1_gen1_1", "gen1_gen2")

# 각 선의 색 지정 (투명도 적용)
colors <- c(
  rgb(1, 0, 0, alpha = 0.8),    # 빨간색
  rgb(0, 0, 1, alpha = 0.8),    # 파란색
  rgb(0, 1, 0, alpha = 0.8),    # 녹색
  rgb(0.5, 0, 0.5, alpha = 0.8) # 보라색
)
# 각 선의 포인트 모양 지정
point_shapes <- c(16, 17, 18, 15)

# 데이터들을 리스트에 정리 (앞서 load한 변수 사용)
datasets <- list(
  ago = list("0.1" = ago_0_1, "0.4" = ago_0_4, "0.7" = ago_0_7, "1.0" = ago_1_0, "1.5" = ago_1_5),
  cnn_news = list("0.1" = cnn_news_0_1, "0.4" = cnn_news_0_4, "0.7" = cnn_news_0_7, "1.0" = cnn_news_1_0, "1.5" = cnn_news_1_5),
  squad2 = list("0.1" = squad_2_0_1, "0.4" = squad_2_0_4, "0.7" = squad_2_0_7, "1.0" = squad_2_1_0, "1.5" = squad_2_1_5),
  quora = list("0.1" = quora_0_1, "0.4" = quora_0_4, "0.7" = quora_0_7, "1.0" = quora_1_0, "1.5" = quora_1_5)
)

# 저장 폴더 경로 (원하는 경로로 수정)
save_path <- "C:/Users/mose/Desktop/minor"
if(!dir.exists(save_path)) dir.create(save_path)

# 각 데이터셋, 각 rho, 그리고 각 유형(열)에 대해 플롯 생성
for(data_name in names(datasets)) {
  for(rho in names(datasets[[data_name]])) {
    # 해당 변수는 7개 요소(각 red_dim별 테이블)를 가진 리스트
    data_list <- datasets[[data_name]][[rho]]
    
    for(col in columns) {
      # 각 테스트에 대해, 해당 유형(col)에서 test 행의 값을 red_dims 순서대로 추출
      # values_mat: 행은 red_dims, 열은 tests (즉, 7 x 4 행렬)
      values_mat <- sapply(tests, function(test) {
        sapply(data_list, function(tbl) {
          as.numeric(tbl[test, col])
        })
      })
      
      # y축 범위를 0~1로 고정
      ylim_range <- c(0, 1)
      
      # 저장할 파일명 지정 (예: plot_ago_or_gen1_1_t.test_2-5_rho_0.1.png)
      file_name <- file.path(save_path, paste0("plot_", data_name, "_", col, "_rho_", rho, ".png"))
      png(filename = file_name, width = 800, height = 600)
      
      # 첫 번째 테스트의 값으로 기본 plot 생성
      plot(red_dims, values_mat[,1], type = "o", col = colors[1],
           pch = point_shapes[1], lty = 1, lwd = 2,
           ylim = ylim_range,
           xlab = "Reduction Dimension", ylab = "Value",
           main = paste(data_name, col, "rho =", rho, "\nTests:", paste(tests, collapse = ", ")))
      
      # 나머지 테스트의 선 추가
      for(i in 2:ncol(values_mat)) {
        lines(red_dims, values_mat[, i], type = "o", col = colors[i],
              pch = point_shapes[i], lty = i, lwd = 2)
      }
      
      # legend 추가 (박스라인 제거)
      legend("topright", legend = tests, col = colors, lty = 1:ncol(values_mat),
             pch = point_shapes, lwd = 2, cex = 0.8, bty = "n")
      
      dev.off()  # 파일 저장 후 장치 종료
    }
  }
}
