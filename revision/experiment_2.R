library(energy)
library(Ball)
library(data.table)

comparison_tests <- function(
    X, 
    Y, 
    source = "quora",        # <= 추가: 데이터 출처 구분 (quora, cnn, squad2, etc.)
    X_name = "X", 
    Y_name = "Y", 
    kfds = 5, 
    method = c("eqdist", "ball", "kernel"), 
    seed = 123
) {
  # 1) 초기 세팅
  set.seed(seed)
  method <- match.arg(method)  # "eqdist", "ball", "kernel" 중 하나 선택
  
  n <- nrow(X)
  if(nrow(X) != nrow(Y)) {
    stop("X와 Y의 행(샘플) 수가 일치해야 합니다.")
  }
  
  # k-폴드 분할
  folds <- split(sample(1:n), rep(1:kfds, length.out = n))
  
  # 결과 저장용 리스트
  res <- list()
  
  # 파일명 생성 함수
  generate_filename <- function(X_name, Y_name, method, seed) {
    paste0(
      "comparison_tests_",
      X_name, "_",
      Y_name, "_",
      method, "_",
      seed, ".txt"
    )
  }
  
  # 최종 출력 파일명
  output_file <- generate_filename(X_name, Y_name, method, seed)
  
  # 2) 테스트 진행 및 결과를 캡쳐
  output_content <- capture.output({
    cat("==== Comparison Tests ====\n")
    cat("Method   :", method, "\n")
    cat("X_name   :", X_name, "\n")
    cat("Y_name   :", Y_name, "\n")
    cat("k-fold   :", kfds, "\n")
    cat("Seed     :", seed, "\n")
    cat("Total N  :", n, "\n")
    cat("Source   :", source, "\n\n")  # <= 추가: source 출력
    
    # folds 인덱스를 가장 먼저 출력
    cat("==== Fold Indices ====\n")
    for (i in seq_len(kfds)) {
      cat("Fold", i, ":", paste(folds[[i]], collapse = ", "), "\n")
    }
    cat("\n==== Test Results ====\n")
    
    # 각 fold별 검정 진행
    for (i in seq_len(kfds)) {
      current_idx <- folds[[i]]
      X_part <- X[current_idx, , drop = FALSE]
      Y_part <- Y[current_idx, , drop = FALSE]
      
      if (method == "eqdist") {
        # energy 패키지 사용
        combined_data <- rbind(X_part, Y_part)
        test_result <- energy::eqdist.etest(
          dist(combined_data), 
          sizes = c(nrow(X_part), nrow(Y_part)), 
          R = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(eqdist) ###\n")
        print(test_result)
        
      } else if (method == "ball") {
        # Ball 패키지 사용
        test_result <- Ball::bd.test(
          X_part, 
          Y_part, 
          num.permutations = 199
        )
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(ball) ###\n")
        print(test_result)
        
      } else if (method == "kernel") {
        # kerTests 패키지 사용
        # sig는 예시로 median heuristic을 이용 (med_sigma 함수 필요)
        sig <- med_sigma(X_part, Y_part)
        test_result <- kerTests::kertests(
          X_part, 
          Y_part, 
          sigma = sig
        )
        res[[paste0("Fold_", i)]] <- test_result
        cat("\n### Fold", i, "(kernel) ###\n")
        print(test_result)
      }
    }
    
  })  # capture.output 종료
  
  # 3) 캡쳐한 출력 내용을 파일로 저장
  writeLines(output_content, con = output_file)
  cat("Saved output to file:", output_file, "\n")
  
  # 4) 결과 리스트 반환
  return(res)
}


'
Load
'


setwd("D:/SVD_data/")

list.files()

# ori
svd_ori_quora <- readRDS("ori_SVD_qr.RDS")
svd_ori_cnn_news <- readRDS("ori_SVD_cnn.RDS")
svd_ori_squad_2 <- readRDS("ori_SVD_sqd.RDS")
  
# Quora
load("svd_gen1_quora_0_1.rds")
svd_gen1_quora_0_1 <- svd_quora

load("svd_gen1_quora_0_4.rds")
svd_gen1_quora_0_4 <- svd_quora

load("svd_gen1_quora_0_7.rds")
svd_gen1_quora_0_7 <- svd_quora

load("svd_gen1_quora_1_0.rds")
svd_gen1_quora_1_0 <- svd_quora

load("svd_gen2_quora_0_1.rds")
svd_gen2_quora_0_1 <- svd_quora

load("svd_gen2_quora_0_4.rds")
svd_gen2_quora_0_4 <- svd_quora

load("svd_gen2_quora_0_7.rds")
svd_gen2_quora_0_7 <- svd_quora

load("svd_gen2_quora_1_0.rds")
svd_gen2_quora_1_0 <- svd_quora

load("svd_gen1_1_quora_0_1.rds")
svd_gen1_1_quora_0_1 <- svd_quora

load("svd_gen1_1_quora_0_4.rds")
svd_gen1_1_quora_0_4 <- svd_quora

load("svd_gen1_1_quora_0_7.rds")
svd_gen1_1_quora_0_7 <- svd_quora

load("svd_gen1_1_quora_1_0.rds")
svd_gen1_1_quora_1_0 <- svd_quora

# CNN
load("svd_gen1_cnn_news_0_1.rds")
svd_gen1_cnn_news_0_1 <- svd_cnn_news

load("svd_gen1_cnn_news_0_4.rds")
svd_gen1_cnn_news_0_4 <- svd_cnn_news

load("svd_gen1_cnn_news_0_7.rds")
svd_gen1_cnn_news_0_7 <- svd_cnn_news

load("svd_gen1_cnn_news_1_0.rds")
svd_gen1_cnn_news_1_0 <- svd_cnn_news

load("svd_gen2_cnn_news_0_1.rds")
svd_gen2_cnn_news_0_1 <- svd_cnn_news

load("svd_gen2_cnn_news_0_4.rds")
svd_gen2_cnn_news_0_4 <- svd_cnn_news

load("svd_gen2_cnn_news_0_7.rds")
svd_gen2_cnn_news_0_7 <- svd_cnn_news

load("svd_gen2_cnn_news_1_0.rds")
svd_gen2_cnn_news_1_0 <- svd_cnn_news

load("svd_gen1_1_cnn_news_0_1.rds")
svd_gen1_1_cnn_news_0_1 <- svd_cnn_news

load("svd_gen1_1_cnn_news_0_4.rds")
svd_gen1_1_cnn_news_0_4 <- svd_cnn_news

load("svd_gen1_1_cnn_news_0_7.rds")
svd_gen1_1_cnn_news_0_7 <- svd_cnn_news

load("svd_gen1_1_cnn_news_1_0.rds")
svd_gen1_1_cnn_news_1_0 <- svd_cnn_news

# Squad2
load("svd_gen1_squad_2_0_1.rds")
svd_gen1_squad_2_0_1 <- svd_squad_2

load("svd_gen1_squad_2_0_4.rds")
svd_gen1_squad_2_0_4 <- svd_squad_2

load("svd_gen1_squad_2_0_7.rds")
svd_gen1_squad_2_0_7 <- svd_squad_2

load("svd_gen1_squad_2_1_0.rds")
svd_gen1_squad_2_1_0 <- svd_squad_2

load("svd_gen2_squad_2_0_1.rds")
svd_gen2_squad_2_0_1 <- svd_squad_2

load("svd_gen2_squad_2_0_4.rds")
svd_gen2_squad_2_0_4 <- svd_squad_2

load("svd_gen2_squad_2_0_7.rds")
svd_gen2_squad_2_0_7 <- svd_squad_2

load("svd_gen2_squad_2_1_0.rds")
svd_gen2_squad_2_1_0 <- svd_squad_2

load("svd_gen1_1_squad_2_0_1.rds")
svd_gen1_1_squad_2_0_1 <- svd_squad_2

load("svd_gen1_1_squad_2_0_4.rds")
svd_gen1_1_squad_2_0_4 <- svd_squad_2

load("svd_gen1_1_squad_2_0_7.rds")
svd_gen1_1_squad_2_0_7 <- svd_squad_2

load("svd_gen1_1_squad_2_1_0.rds")
svd_gen1_1_squad_2_1_0 <- svd_squad_2

'
reduction
'

###############################################################################
# 1. 설정
###############################################################################
dim <- 3 # 축소 차원
orig_dim <- 1536  # 원본 차원

###############################################################################
# 2. Original SVD (ori_SVD_qr, ori_SVD_cnn, ori_SVD_sqd)
###############################################################################
red_ori_quora <- svd_ori_quora$u %*% rbind(
  diag(svd_ori_quora$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_ori_cnn_news <- svd_ori_cnn_news$u %*% rbind(
  diag(svd_ori_cnn_news$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_ori_squad_2 <- svd_ori_squad_2$u %*% rbind(
  diag(svd_ori_squad_2$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

###############################################################################
# 3. Quora 계열 SVD
###############################################################################
# gen1
red_gen1_quora_0_1 <- svd_gen1_quora_0_1$u %*% rbind(
  diag(svd_gen1_quora_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_0_4 <- svd_gen1_quora_0_4$u %*% rbind(
  diag(svd_gen1_quora_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_0_7 <- svd_gen1_quora_0_7$u %*% rbind(
  diag(svd_gen1_quora_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_quora_1_0 <- svd_gen1_quora_1_0$u %*% rbind(
  diag(svd_gen1_quora_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen2
red_gen2_quora_0_1 <- svd_gen2_quora_0_1$u %*% rbind(
  diag(svd_gen2_quora_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_quora_0_4 <- svd_gen2_quora_0_4$u %*% rbind(
  diag(svd_gen2_quora_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_quora_0_7 <- svd_gen2_quora_0_7$u %*% rbind(
  diag(svd_gen2_quora_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_quora_1_0 <- svd_gen2_quora_1_0$u %*% rbind(
  diag(svd_gen2_quora_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen1_1
red_gen1_1_quora_0_1 <- svd_gen1_1_quora_0_1$u %*% rbind(
  diag(svd_gen1_1_quora_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_quora_0_4 <- svd_gen1_1_quora_0_4$u %*% rbind(
  diag(svd_gen1_1_quora_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_quora_0_7 <- svd_gen1_1_quora_0_7$u %*% rbind(
  diag(svd_gen1_1_quora_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_quora_1_0 <- svd_gen1_1_quora_1_0$u %*% rbind(
  diag(svd_gen1_1_quora_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

###############################################################################
# 4. CNN 계열 SVD
###############################################################################
# gen1
red_gen1_cnn_news_0_1 <- svd_gen1_cnn_news_0_1$u %*% rbind(
  diag(svd_gen1_cnn_news_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_0_4 <- svd_gen1_cnn_news_0_4$u %*% rbind(
  diag(svd_gen1_cnn_news_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_0_7 <- svd_gen1_cnn_news_0_7$u %*% rbind(
  diag(svd_gen1_cnn_news_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_cnn_news_1_0 <- svd_gen1_cnn_news_1_0$u %*% rbind(
  diag(svd_gen1_cnn_news_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen2
red_gen2_cnn_news_0_1 <- svd_gen2_cnn_news_0_1$u %*% rbind(
  diag(svd_gen2_cnn_news_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_cnn_news_0_4 <- svd_gen2_cnn_news_0_4$u %*% rbind(
  diag(svd_gen2_cnn_news_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_cnn_news_0_7 <- svd_gen2_cnn_news_0_7$u %*% rbind(
  diag(svd_gen2_cnn_news_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_cnn_news_1_0 <- svd_gen2_cnn_news_1_0$u %*% rbind(
  diag(svd_gen2_cnn_news_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen1_1
red_gen1_1_cnn_news_0_1 <- svd_gen1_1_cnn_news_0_1$u %*% rbind(
  diag(svd_gen1_1_cnn_news_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_cnn_news_0_4 <- svd_gen1_1_cnn_news_0_4$u %*% rbind(
  diag(svd_gen1_1_cnn_news_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_cnn_news_0_7 <- svd_gen1_1_cnn_news_0_7$u %*% rbind(
  diag(svd_gen1_1_cnn_news_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_cnn_news_1_0 <- svd_gen1_1_cnn_news_1_0$u %*% rbind(
  diag(svd_gen1_1_cnn_news_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

###############################################################################
# 5. Squad2 계열 SVD
###############################################################################
# gen1
red_gen1_squad_2_0_1 <- svd_gen1_squad_2_0_1$u %*% rbind(
  diag(svd_gen1_squad_2_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_0_4 <- svd_gen1_squad_2_0_4$u %*% rbind(
  diag(svd_gen1_squad_2_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_0_7 <- svd_gen1_squad_2_0_7$u %*% rbind(
  diag(svd_gen1_squad_2_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_squad_2_1_0 <- svd_gen1_squad_2_1_0$u %*% rbind(
  diag(svd_gen1_squad_2_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen2
red_gen2_squad_2_0_1 <- svd_gen2_squad_2_0_1$u %*% rbind(
  diag(svd_gen2_squad_2_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_squad_2_0_4 <- svd_gen2_squad_2_0_4$u %*% rbind(
  diag(svd_gen2_squad_2_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_squad_2_0_7 <- svd_gen2_squad_2_0_7$u %*% rbind(
  diag(svd_gen2_squad_2_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen2_squad_2_1_0 <- svd_gen2_squad_2_1_0$u %*% rbind(
  diag(svd_gen2_squad_2_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

# gen1_1
red_gen1_1_squad_2_0_1 <- svd_gen1_1_squad_2_0_1$u %*% rbind(
  diag(svd_gen1_1_squad_2_0_1$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_squad_2_0_4 <- svd_gen1_1_squad_2_0_4$u %*% rbind(
  diag(svd_gen1_1_squad_2_0_4$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_squad_2_0_7 <- svd_gen1_1_squad_2_0_7$u %*% rbind(
  diag(svd_gen1_1_squad_2_0_7$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)

red_gen1_1_squad_2_1_0 <- svd_gen1_1_squad_2_1_0$u %*% rbind(
  diag(svd_gen1_1_squad_2_1_0$d[1:dim]),
  matrix(0, nrow = orig_dim - dim, ncol = dim)
)


'
testing
'

setwd("D:/testing/")

'
temp 0.1
'

## eqdist
# quora (5개: za ~ ze) => a ~ e
a <- comparison_tests(
  X = red_gen1_quora_0_1, 
  Y = red_gen2_quora_0_1, 
  source = 'quora',
  X_name = 'gen1_quora_0_1', 
  Y_name = 'gen2_quora_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

b <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

c <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

d <- comparison_tests(
  X = red_gen1_quora_0_1, 
  Y = red_gen1_1_quora_0_1, 
  source = 'quora',
  X_name = 'gen1_quora_0_1', 
  Y_name = 'gen1_1_quora_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

e <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)



# squad_2 (5개: zf ~ zj) => f ~ j
f <- comparison_tests(
  X = red_gen1_squad_2_0_1, 
  Y = red_gen2_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_1', 
  Y_name = 'gen2_squad_2_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

g <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

h <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

i <- comparison_tests(
  X = red_gen1_squad_2_0_1, 
  Y = red_gen1_1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_1', 
  Y_name = 'gen1_1_squad_2_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

j <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)



# cnn_news (5개: zk ~ z0) => k ~ o
k <- comparison_tests(
  X = red_gen1_cnn_news_0_1, 
  Y = red_gen2_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_1', 
  Y_name = 'gen2_cnn_news_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

l <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

m <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

n <- comparison_tests(
  X = red_gen1_cnn_news_0_1, 
  Y = red_gen1_1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_1', 
  Y_name = 'gen1_1_cnn_news_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)

o <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_1',
  kfds = 5, method = 'eqdist', seed = 7
)



## ball
# quora (5개: a, b, c, fdafs, dsacsd) => p ~ t
p <- comparison_tests(
  X = red_gen1_quora_0_1, 
  Y = red_gen2_quora_0_1, 
  source = 'quora',
  X_name = 'gen1_quora_0_1', 
  Y_name = 'gen2_quora_0_1',
  kfds = 5, method = 'ball', seed = 7
)

q <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_1',
  kfds = 5, method = 'ball', seed = 7
)

r <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_1',
  kfds = 5, method = 'ball', seed = 7
)

s <- comparison_tests(
  X = red_gen1_quora_0_1, 
  Y = red_gen1_1_quora_0_1, 
  source = 'quora',
  X_name = 'gen1_quora_0_1', 
  Y_name = 'gen1_1_quora_0_1',
  kfds = 5, method = 'ball', seed = 7
)

t <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_1, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_1',
  kfds = 5, method = 'ball', seed = 7
)



# squad_2 (5개: cvv, asdac, dsafsv, asdfsda, vzvcx) => u ~ y
u <- comparison_tests(
  X = red_gen1_squad_2_0_1, 
  Y = red_gen2_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_1', 
  Y_name = 'gen2_squad_2_0_1',
  kfds = 5, method = 'ball', seed = 7
)

v <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_1',
  kfds = 5, method = 'ball', seed = 7
)

w <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_1',
  kfds = 5, method = 'ball', seed = 7
)

x <- comparison_tests(
  X = red_gen1_squad_2_0_1, 
  Y = red_gen1_1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_1', 
  Y_name = 'gen1_1_squad_2_0_1',
  kfds = 5, method = 'ball', seed = 7
)

y <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_1, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_1',
  kfds = 5, method = 'ball', seed = 7
)

# cnn_news (5개: acsdacdcacs, zxcvxvcv, qweqwe, sdafewgdv, qwcxc) => z ~ ad
z <- comparison_tests(
  X = red_gen1_cnn_news_0_1, 
  Y = red_gen2_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_1', 
  Y_name = 'gen2_cnn_news_0_1',
  kfds = 5, method = 'ball', seed = 7
)

aa <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_1',
  kfds = 5, method = 'ball', seed = 7
)

ab <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_1',
  kfds = 5, method = 'ball', seed = 7
)

ac <- comparison_tests(
  X = red_gen1_cnn_news_0_1, 
  Y = red_gen1_1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_1', 
  Y_name = 'gen1_1_cnn_news_0_1',
  kfds = 5, method = 'ball', seed = 7
)

ad <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_1, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_1',
  kfds = 5, method = 'ball', seed = 7
)


'
temp 0.4
'
## eqdist
# quora (5개: qp, qeqdcv, qwewqsa, abcd, efg) => ae ~ ai
ae <- comparison_tests(
  X = red_gen1_quora_0_4, 
  Y = red_gen2_quora_0_4, 
  source = 'quora',
  X_name = 'gen1_quora_0_4', 
  Y_name = 'gen2_quora_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

af <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ag <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ah <- comparison_tests(
  X = red_gen1_quora_0_4, 
  Y = red_gen1_1_quora_0_4, 
  source = 'quora',
  X_name = 'gen1_quora_0_4', 
  Y_name = 'gen1_1_quora_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ai <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)


# squad_2 (5개: hic, klml, qsc, zxpd, afdsdfdfvbn) => aj ~ an
aj <- comparison_tests(
  X = red_gen1_squad_2_0_4, 
  Y = red_gen2_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_4', 
  Y_name = 'gen2_squad_2_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ak <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

al <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

am <- comparison_tests(
  X = red_gen1_squad_2_0_4, 
  Y = red_gen1_1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_4', 
  Y_name = 'gen1_1_squad_2_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

an <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)


# cnn_news (5개: agj, pms, crkfdsafsdvc, cxzvxvasvawdvdv, dasvsvcsavac) => ao ~ as
ao <- comparison_tests(
  X = red_gen1_cnn_news_0_4, 
  Y = red_gen2_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_4', 
  Y_name = 'gen2_cnn_news_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ap <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

aq <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

ar <- comparison_tests(
  X = red_gen1_cnn_news_0_4, 
  Y = red_gen1_1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_4', 
  Y_name = 'gen1_1_cnn_news_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)

as <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_4',
  kfds = 5, method = 'eqdist', seed = 7
)


# ball
# quora (5개: sdfqfqc, tmtmdh, ndfbf, cvfwvfvw, wfvvwfe) => at ~ ax
at <- comparison_tests(
  X = red_gen1_quora_0_4, 
  Y = red_gen2_quora_0_4, 
  source = 'quora',
  X_name = 'gen1_quora_0_4', 
  Y_name = 'gen2_quora_0_4',
  kfds = 5, method = 'ball', seed = 7
)

au <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_4',
  kfds = 5, method = 'ball', seed = 7
)

av <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_4',
  kfds = 5, method = 'ball', seed = 7
)

aw <- comparison_tests(
  X = red_gen1_quora_0_4, 
  Y = red_gen1_1_quora_0_4, 
  source = 'quora',
  X_name = 'gen1_quora_0_4', 
  Y_name = 'gen1_1_quora_0_4',
  kfds = 5, method = 'ball', seed = 7
)

ax <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_4, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_4',
  kfds = 5, method = 'ball', seed = 7
)

# squad_2 (5개: cdaCxsc, sxcasv, scvascvbfb, bgbgbggbbg, asaas) => ay ~ bc
ay <- comparison_tests(
  X = red_gen1_squad_2_0_4, 
  Y = red_gen2_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_4', 
  Y_name = 'gen2_squad_2_0_4',
  kfds = 5, method = 'ball', seed = 7
)

az <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_4',
  kfds = 5, method = 'ball', seed = 7
)

ba <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_4',
  kfds = 5, method = 'ball', seed = 7
)

bb <- comparison_tests(
  X = red_gen1_squad_2_0_4, 
  Y = red_gen1_1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_4', 
  Y_name = 'gen1_1_squad_2_0_4',
  kfds = 5, method = 'ball', seed = 7
)

bc <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_4, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_4',
  kfds = 5, method = 'ball', seed = 7
)


# cnn_news (5개: cdcacdcsdeqfcq, csacac3, xcvxcvsvd4, dsaf1, abc13ds) => bd ~ bh
bd <- comparison_tests(
  X = red_gen1_cnn_news_0_4, 
  Y = red_gen2_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_4', 
  Y_name = 'gen2_cnn_news_0_4',
  kfds = 5, method = 'ball', seed = 7
)

be <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_4',
  kfds = 5, method = 'ball', seed = 7
)

bf <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_4',
  kfds = 5, method = 'ball', seed = 7
)

bg <- comparison_tests(
  X = red_gen1_cnn_news_0_4, 
  Y = red_gen1_1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_4', 
  Y_name = 'gen1_1_cnn_news_0_4',
  kfds = 5, method = 'ball', seed = 7
)

bh <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_4, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_4',
  kfds = 5, method = 'ball', seed = 7
)

'
temp 0.7
'

## eqdist
# quora (5개: k, l, m, n, o) => bi ~ bm
bi <- comparison_tests(
  X = red_gen1_quora_0_7, 
  Y = red_gen2_quora_0_7, 
  source = 'quora',
  X_name = 'gen1_quora_0_7', 
  Y_name = 'gen2_quora_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bj <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bk <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bl <- comparison_tests(
  X = red_gen1_quora_0_7, 
  Y = red_gen1_1_quora_0_7, 
  source = 'quora',
  X_name = 'gen1_quora_0_7', 
  Y_name = 'gen1_1_quora_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bm <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)


# squad_2 (5개: p, q, r, s, t) => bn ~ br
bn <- comparison_tests(
  X = red_gen1_squad_2_0_7, 
  Y = red_gen2_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_7', 
  Y_name = 'gen2_squad_2_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bo <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bp <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bq <- comparison_tests(
  X = red_gen1_squad_2_0_7, 
  Y = red_gen1_1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_7', 
  Y_name = 'gen1_1_squad_2_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

br <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)


# cnn_news (5개: u, v, w, x, y) => bs ~ bw
bs <- comparison_tests(
  X = red_gen1_cnn_news_0_7, 
  Y = red_gen2_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_7', 
  Y_name = 'gen2_cnn_news_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bt <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bu <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bv <- comparison_tests(
  X = red_gen1_cnn_news_0_7, 
  Y = red_gen1_1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_7', 
  Y_name = 'gen1_1_cnn_news_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)

bw <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_7',
  kfds = 5, method = 'eqdist', seed = 7
)


## ball
# quora (5개: z, aa, ab, ac, ad) => bx ~ cb
bx <- comparison_tests(
  X = red_gen1_quora_0_7, 
  Y = red_gen2_quora_0_7, 
  source = 'quora',
  X_name = 'gen1_quora_0_7', 
  Y_name = 'gen2_quora_0_7',
  kfds = 5, method = 'ball', seed = 7
)

by <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_0_7',
  kfds = 5, method = 'ball', seed = 7
)

bz <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_0_7',
  kfds = 5, method = 'ball', seed = 7
)

ca <- comparison_tests(
  X = red_gen1_quora_0_7, 
  Y = red_gen1_1_quora_0_7, 
  source = 'quora',
  X_name = 'gen1_quora_0_7', 
  Y_name = 'gen1_1_quora_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cb <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_0_7, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_0_7',
  kfds = 5, method = 'ball', seed = 7
)


# squad_2 (5개: ae, af, ag, ah, ai) => cc ~ cg
cc <- comparison_tests(
  X = red_gen1_squad_2_0_7, 
  Y = red_gen2_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_7', 
  Y_name = 'gen2_squad_2_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cd <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_0_7',
  kfds = 5, method = 'ball', seed = 7
)

ce <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cf <- comparison_tests(
  X = red_gen1_squad_2_0_7, 
  Y = red_gen1_1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_0_7', 
  Y_name = 'gen1_1_squad_2_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cg <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_0_7, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_0_7',
  kfds = 5, method = 'ball', seed = 7
)


# cnn_news (5개: aj, ak, al, am, an) => ch ~ cl
ch <- comparison_tests(
  X = red_gen1_cnn_news_0_7, 
  Y = red_gen2_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_7', 
  Y_name = 'gen2_cnn_news_0_7',
  kfds = 5, method = 'ball', seed = 7
)

ci <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cj <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_0_7',
  kfds = 5, method = 'ball', seed = 7
)

ck <- comparison_tests(
  X = red_gen1_cnn_news_0_7, 
  Y = red_gen1_1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_0_7', 
  Y_name = 'gen1_1_cnn_news_0_7',
  kfds = 5, method = 'ball', seed = 7
)

cl <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_0_7, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_0_7',
  kfds = 5, method = 'ball', seed = 7
)


'
temp 1.0
'

## eqdist
# quora (5개: ao, ap, aq, ar, as) => cm ~ cq
cm <- comparison_tests(
  X = red_gen1_quora_1_0, 
  Y = red_gen2_quora_1_0, 
  source = 'quora',
  X_name = 'gen1_quora_1_0', 
  Y_name = 'gen2_quora_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cn <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

co <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cp <- comparison_tests(
  X = red_gen1_quora_1_0, 
  Y = red_gen1_1_quora_1_0, 
  source = 'quora',
  X_name = 'gen1_quora_1_0', 
  Y_name = 'gen1_1_quora_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cq <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)


# squad_2 (5개: at, au, av, aw, ax) => cr ~ cv
cr <- comparison_tests(
  X = red_gen1_squad_2_1_0, 
  Y = red_gen2_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_1_0', 
  Y_name = 'gen2_squad_2_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cs <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

ct <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cu <- comparison_tests(
  X = red_gen1_squad_2_1_0, 
  Y = red_gen1_1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_1_0', 
  Y_name = 'gen1_1_squad_2_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cv <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)


# cnn_news (5개: ay, az, ba, bb, bc) => cw ~ da
cw <- comparison_tests(
  X = red_gen1_cnn_news_1_0, 
  Y = red_gen2_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_1_0', 
  Y_name = 'gen2_cnn_news_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cx <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cy <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

cz <- comparison_tests(
  X = red_gen1_cnn_news_1_0, 
  Y = red_gen1_1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_1_0', 
  Y_name = 'gen1_1_cnn_news_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)

da <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_1_0',
  kfds = 5, method = 'eqdist', seed = 7
)


## ball
# quora (5개: bs, bt, bu, bv, bw) => db ~ df
db <- comparison_tests(
  X = red_gen1_quora_1_0, 
  Y = red_gen2_quora_1_0, 
  source = 'quora',
  X_name = 'gen1_quora_1_0', 
  Y_name = 'gen2_quora_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dc <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen2_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen2_quora_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dd <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_quora_1_0',
  kfds = 5, method = 'ball', seed = 7
)

de <- comparison_tests(
  X = red_gen1_quora_1_0, 
  Y = red_gen1_1_quora_1_0, 
  source = 'quora',
  X_name = 'gen1_quora_1_0', 
  Y_name = 'gen1_1_quora_1_0',
  kfds = 5, method = 'ball', seed = 7
)

df <- comparison_tests(
  X = red_ori_quora, 
  Y = red_gen1_1_quora_1_0, 
  source = 'quora',
  X_name = 'ori_quora', 
  Y_name = 'gen1_1_quora_1_0',
  kfds = 5, method = 'ball', seed = 7
)


# squad_2 (5개: bx, by, bz, ca, cb) => dg ~ dk
dg <- comparison_tests(
  X = red_gen1_squad_2_1_0, 
  Y = red_gen2_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_1_0', 
  Y_name = 'gen2_squad_2_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dh <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen2_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen2_squad_2_1_0',
  kfds = 5, method = 'ball', seed = 7
)

di <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_squad_2_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dj <- comparison_tests(
  X = red_gen1_squad_2_1_0, 
  Y = red_gen1_1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'gen1_squad_2_1_0', 
  Y_name = 'gen1_1_squad_2_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dk <- comparison_tests(
  X = red_ori_squad_2, 
  Y = red_gen1_1_squad_2_1_0, 
  source = 'squad_2',
  X_name = 'ori_squad_2', 
  Y_name = 'gen1_1_squad_2_1_0',
  kfds = 5, method = 'ball', seed = 7
)


# cnn_news (5개: cc, cd, ce, cf, cg) => dl ~ dp
dl <- comparison_tests(
  X = red_gen1_cnn_news_1_0, 
  Y = red_gen2_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_1_0', 
  Y_name = 'gen2_cnn_news_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dm <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen2_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen2_cnn_news_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dn <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_cnn_news_1_0',
  kfds = 5, method = 'ball', seed = 7
)

do <- comparison_tests(
  X = red_gen1_cnn_news_1_0, 
  Y = red_gen1_1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'gen1_cnn_news_1_0', 
  Y_name = 'gen1_1_cnn_news_1_0',
  kfds = 5, method = 'ball', seed = 7
)

dp <- comparison_tests(
  X = red_ori_cnn_news, 
  Y = red_gen1_1_cnn_news_1_0, 
  source = 'cnn_news',
  X_name = 'ori_cnn_news', 
  Y_name = 'gen1_1_cnn_news_1_0',
  kfds = 5, method = 'ball', seed = 7
)
