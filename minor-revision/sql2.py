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
