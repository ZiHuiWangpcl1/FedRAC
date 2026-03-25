# 选择基础镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 复制当前目录所有文件到容器
COPY . .

# 安装所需的 Python 库
RUN pip install -r requirements.txt

# 设置默认启动命令
CMD ["python", "app.py"]