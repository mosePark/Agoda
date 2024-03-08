FROM python:3.9

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install scikit-build cmake ninja

# GOSDT 라이브러리 복사 (혹은 git clone 등 다른 방법으로 가져오기)
COPY . /gosdt
WORKDIR /gosdt

# 라이브러리 빌드 및 설치 스크립트 실행
RUN python build.py
