import torch
import torch.nn.functional as F


# 函数式自定义损失函数
def focal_loss(predictions, targets, gamma=2., alpha=0.25):
    """
    Focal Loss: 解决目标检测中正负样本严重不平衡的问题
    """
    ce_loss = F.cross_entropy(predictions, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # 预测正确的概率
    focal_loss = alpha * (1 - pt) ** gamma  # 难样本权重更高
    return (focal_loss * ce_loss).mean()


# 使用
predictions = torch.randn(8, 10)
targets = torch.randn(0, 10, (8,))
loss = focal_loss(predictions, targets)
