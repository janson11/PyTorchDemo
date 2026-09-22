import json

# ResNet18 ImageNet 1000分类，简单索引映射
index_to_name = {str(i): f"class_{i}" for i in range(1, 1000)}

with open("index_to_name.json", "w", encoding="utf-8") as f:
    json.dump(index_to_name, f)