import torch
import torch.nn as nn


# 方式二 继承nn.Module(推荐)
class DiceLoss(nn.Module):
    """
    Dice Loss：常用于图像分割，直接优化 Dice 系数
    对类别不平衡（如小目标分割）比 CrossEntropy 更鲁棒
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        # predictions: (N, C, H, W) -> 经过 sigmoid 的概率
        # targets:     (N, C, H, W) -> one-hot 编码标签
        predictions = torch.sigmoid(predictions)
        # 展平道(N,-1)
        pred_flat = predictions.view(predictions.size(0), -1)
        target_flat = targets.view(targets.size(0), -1).float()
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
                pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    组合损失：CrossEntropy + Dice，兼顾像素级分类和区域重叠
    图像分割常用组合
    """

    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        return (self.ce_weight * self.ce(predictions, targets) +
                self.dice_weight * self.dice(predictions, targets))

# 使用
criterion = CombinedLoss(ce_weight=0.4, dice_weight=0.6)
