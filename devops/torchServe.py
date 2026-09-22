# 安装
# pip install torchserve torch-model-archiver

# 打包模型
"""
bash

torch-model-archiver --model-name resnet18 \
--version 1.0 \
--serialized-file best_model.pth \
--extra-files index_to_name.json \
--handler image_classifier \
--export-path model_store

"""


# 启动服务
# torchserve --start --model-store model_store --models resnet18=resnet18.mar
