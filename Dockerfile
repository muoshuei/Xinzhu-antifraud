# Dockerfile

# =================================================================
# 階段 1: BUILDER 階段 (安裝所有大型 ML 庫)
# 使用一個帶有編譯器依賴的 python:slim 映像
FROM python:3.11-slim as builder 

# 設定安裝目錄
WORKDIR /install

# --- 安裝系統依賴 (確保像 numpy/scikit-learn 能夠順利編譯) ---
# 注意：這些系統依賴會在這裡被清除，不會進入最終映像
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gfortran \
    libblas-dev \
    liblapack-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- 安裝大型 Python 依賴 ---
COPY requirements_base.txt .
# 使用 --no-cache-dir 確保安裝過程不會產生不必要的快取檔案
RUN pip install --no-cache-dir -r requirements_base.txt


# =================================================================
# 階段 2: FINAL 運行階段 (極簡化，適用於 Cloud Run)
# -----------------------------------------------------------------
# 使用相同的 python:slim 映像作為運行時環境 (極小化)
FROM python:3.11-slim

# 設定應用程式工作目錄
WORKDIR /app

# ***關鍵步驟***: 從 BUILDER 階段複製安裝好的 Python 函式庫
# 複製的只有編譯/安裝好的函式庫，不包含系統編譯器和建構時的垃圾
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# --- 安裝輕量級應用程式依賴 ---
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# --- 複製應用程式程式碼 ---
COPY . .

# 運行 FastAPI 應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]