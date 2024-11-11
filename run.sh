#!/bin/bash

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 数据收集
python src/data.py collect_data data/processed

# 模型训练
python src/model.py train_model

# 模型推理
python src/inference.py run_inference models/final_model.pt