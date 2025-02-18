rm(list = ls())

# 작업 디렉토리 설정 및 데이터 불러오기
setwd("C:/Users/mose/Desktop/results_comp")

rdata_files <- list.files(pattern = "\\.[Rr]data$", ignore.case = TRUE)

comp <- read.csv("comp.csv") # 구글시트에 있는 energy, ball 데이터

# RData 파일들 로드
load("agora_0_1.RData"); ago_0_1 <- result_list
load("agora_0_4.RData"); ago_0_4 <- result_list
load("agora_0_7.RData"); ago_0_7 <- result_list
load("agora_1_0.RData"); ago_1_0 <- result_list
load("agora_1_5.RData"); ago_1_5 <- result_list

load("cnn_news_0_1.RData"); cnn_news_0_1 <- result_list
load("cnn_news_0_4.RData"); cnn_news_0_4 <- result_list
load("cnn_news_0_7.RData"); cnn_news_0_7 <- result_list
load("cnn_news_1_0.RData"); cnn_news_1_0 <- result_list
load("cnn_news_1_5.RData"); cnn_news_1_5 <- result_list

load("quora_0_1.RData"); quora_0_1 <- result_list
load("quora_0_4.RData"); quora_0_4 <- result_list
load("quora_0_7.RData"); quora_0_7 <- result_list
load("quora_1_0.RData"); quora_1_0 <- result_list
load("quora_1_5.RData"); quora_1_5 <- result_list

load("squad_2_0_1.RData"); squad_2_0_1 <- result_list
load("squad_2_0_4.RData"); squad_2_0_4 <- result_list
load("squad_2_0_7.RData"); squad_2_0_7 <- result_list
load("squad_2_1_0.RData"); squad_2_1_0 <- result_list
load("squad_2_1_5.RData"); squad_2_1_5 <- result_list

# ---------------------------------------------------
# RData 객체들을 리스트로 정리 (키는 rho 값; 파일명 뒷부분과 동일)
data_rdata <- list(
  ago = list("0_1" = ago_0_1, "0_4" = ago_0_4, "0_7" = ago_0_7, "1_0" = ago_1_0, "1_5" = ago_1_5),
  cnn_news = list("0_1" = cnn_news_0_1, "0_4" = cnn_news_0_4, "0_7" = cnn_news_0_7, "1_0" = cnn_news_1_0, "1_5" = cnn_news_1_5),
  quora = list("0_1" = quora_0_1, "0_4" = quora_0_4, "0_7" = quora_0_7, "1_0" = quora_1_0, "1_5" = quora_1_5),
  squad = list("0_1" = squad_2_0_1, "0_4" = squad_2_0_4, "0_7" = squad_2_0_7, "1_0" = squad_2_1_0, "1_5" = squad_2_1_5)
)

# comp 파일에서 데이터셋 이름과 comp의 data 컬럼 값 매핑 (예: Review, CNN, Quora, SQUAD2)
data_map <- list(
  ago      = "Review",
  cnn_news = "CNN",
  quora    = "Quora",
  squad    = "SQUAD2"
)

# 사용할 rho 값들 (comp 파일의 rho 컬럼도 "0_1", "0_4" 등으로 표기)
rho_vals <- c("0_1", "0_4", "0_7", "1_0", "1_5")

# comp 파일의 type 값들 (예: "or_gen1_1", "or_gen2", "gen1_gen1_1", "gen1_gen2")
type_vals <- c("or_gen1_1", "or_gen2", "gen1_gen1_1", "gen1_gen2")

# 기본적으로 RData 객체의 행 이름은 dim 값 (예: 2, 3, 4, 5, 10, 15, 20)입니다.
default_dims <- c(2, 3, 4, 5, 10, 15, 20)

# 모든 그래프의 y축 범위를 고정 (0 ~ 1)
fixed_ylim <- c(0, 1)

# 결과 파일을 저장할 디렉토리가 없다면 생성
if(!dir.exists("plots")){
  dir.create("plots")
}

# 색상 설정 (투명도 적용)
col_eq   <- rgb(0, 0, 1, alpha = 0.7)    # 파란색: comp eqtest
col_ball <- rgb(1, 0, 0, alpha = 0.7)      # 빨간색: comp balltest
col_hot  <- rgb(0, 0.6, 0, alpha = 0.7)    # 녹색: RData hotelling
col_np   <- rgb(0.5, 0, 0.5, alpha = 0.7)  # 보라색: RData np

# 각 플롯을 PNG 파일로 저장하고, 사용된 데이터 테이블을 콘솔에 출력
for (dset in names(data_rdata)) {
  for (rho in rho_vals) {
    for (tp in type_vals) {
      
      ## 1. comp 데이터에서 해당 데이터셋, rho, type에 해당하는 행 추출
      comp_subset <- comp[ comp$data == data_map[[dset]] & comp$rho == rho & comp$type == tp, ]
      
      # 만약 comp에 해당하는 데이터가 없다면, 기본 dims와 NA 벡터 사용
      if(nrow(comp_subset) == 0){
        x_comp <- default_dims
        eqtest_vals <- rep(NA, length(default_dims))
        balltest_vals <- rep(NA, length(default_dims))
      } else {
        comp_subset <- comp_subset[ order(comp_subset$dim), ]
        x_comp <- as.numeric(comp_subset$dim)  # comp 파일에 기록된 dim 값 사용
        eqtest_vals <- as.numeric(comp_subset$eqtest)
        balltest_vals <- as.numeric(comp_subset$balltest)
      }
      
      ## 2. RData 객체에서 해당 데이터셋과 rho에 해당하는 결과 추출  
      rdata_obj <- data_rdata[[dset]][[rho]]
      # RData 객체에서는 행 이름이 dim 값입니다.
      rdata_dims <- as.numeric(rownames(rdata_obj$hotelling))
      
      # comp의 type (tp) 값을 사용하여 컬럼 선택 (없으면 NA)
      if(tp %in% colnames(rdata_obj$hotelling)){
        hotelling_vals <- as.numeric(rdata_obj$hotelling[ , tp])
      } else {
        hotelling_vals <- rep(NA, length(rdata_dims))
      }
      if(tp %in% colnames(rdata_obj$np)){
        np_vals <- as.numeric(rdata_obj$np[ , tp])
      } else {
        np_vals <- rep(NA, length(rdata_dims))
      }
      
      ## 3. 두 데이터셋의 차원(= x 값) 합집합을 구하고, 각 값 채우기
      dims_union <- sort(unique(c(x_comp, rdata_dims)))
      comp_eq <- rep(NA, length(dims_union))
      comp_ball <- rep(NA, length(dims_union))
      rdata_hot <- rep(NA, length(dims_union))
      rdata_np <- rep(NA, length(dims_union))
      
      # comp 데이터 값 채우기 (중복된 dim은 첫번째 값 사용)
      for(i in seq_along(dims_union)){
        d <- dims_union[i]
        if(d %in% x_comp) {
          idx <- which(x_comp == d)[1]
          comp_eq[i] <- eqtest_vals[idx]
          comp_ball[i] <- balltest_vals[idx]
        }
      }
      # RData 값 채우기
      for(i in seq_along(dims_union)){
        d <- dims_union[i]
        if(d %in% rdata_dims) {
          idx <- which(rdata_dims == d)[1]
          rdata_hot[i] <- hotelling_vals[idx]
          rdata_np[i] <- np_vals[idx]
        }
      }
      
      # 테이블 생성 (합쳐진 데이터)
      table_df <- data.frame(
        Dimension = dims_union,
        comp_eqtest = comp_eq,
        comp_balltest = comp_ball,
        rdata_hotelling = rdata_hot,
        rdata_np = rdata_np
      )
      
      # 콘솔에 데이터프레임 출력
      cat("\n============================================\n")
      cat(paste(data_map[[dset]], "rho =", gsub("_", ".", rho), "type =", tp), "\n")
      print(table_df)
      cat("============================================\n")
      
      ## 4. y축 범위 고정 (0 ~ 1)
      y_range <- fixed_ylim
      
      ## 5. 저장할 파일명 설정 (plots 폴더 내에 저장; type 포함)
      base_filename <- paste0(data_map[[dset]], "_", tp, "_rho_", gsub("_", ".", rho))
      png_file <- paste0("plots/", base_filename, ".png")
      
      ## 6. PNG 디바이스 열기
      png(filename = png_file, width = 800, height = 600)
      
      ## 7. Plot 작성  
      # comp의 eqtest: 실선, 파란색, 점 모양: 원형(16)
      plot(x_comp, eqtest_vals, type = "b", pch = 16, col = col_eq,
           lty = 1, lwd = 2, ylim = y_range,
           xlab = "Dimension", ylab = "Test Value",
           main = paste(data_map[[dset]], "rho =", gsub("_", ".", rho), "type =", tp))
      
      # comp의 balltest: 점선, 빨간색, 점 모양: 다이아몬드(18)
      lines(x_comp, balltest_vals, type = "b", pch = 18, col = col_ball,
            lty = 2, lwd = 2)
      
      # RData의 hotelling: 점선, 녹색, 점 모양: 삼각형(17)
      lines(rdata_dims, hotelling_vals, type = "b", pch = 17, col = col_hot,
            lty = 3, lwd = 2)
      
      # RData의 np: 점선, 보라색, 점 모양: 정사각형(15)
      lines(rdata_dims, np_vals, type = "b", pch = 15, col = col_np,
            lty = 4, lwd = 2)
      
      # 범례 추가
      legend("topright", legend = c("eqtest", "balltest", "hotelling", "Nploc"),
             col = c(col_eq, col_ball, col_hot, col_np),
             lty = c(1, 2, 3, 4), pch = c(16, 18, 17, 15), lwd = 2, cex = 0.8)
      
      ## 8. PNG 디바이스 닫기
      dev.off()
    }
  }
}
