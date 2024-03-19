#!/bin/bash

# 컨테이너 이름 설정
CONTAINER_NAME="gosdt-container"

# 사용할 이미지의 경로. Docker Hub에서 가져올 수도 있고, 로컬 파일 시스템의 경로일 수도 있습니다.
CONTAINER_IMAGE_PATH="docker://achreto/gosdt"

# 컨테이너 이미지를 저장할 로컬 경로
CONTAINER_PATH="$HOME/agoda/image/achreto/gosdt.sqsh"
