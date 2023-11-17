# SOME/IP IDS
## 1. 环境搭建
### 1.1. 安装 anaconda/miniconda
略.
### 1.2. 安装包
依次输入以下指令:
```bash
conda create -n someip_ids python=3.10
conda activate someip_ids
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install tensorflow
```
## 2. 运行与测试
运行 [main.py](./main.py) 即可.