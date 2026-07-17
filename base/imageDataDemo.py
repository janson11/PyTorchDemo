import torchvision.transforms as transforms
from PIL import Image

# 定义数据预处理的流水线
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),# 随机水平翻转
    transforms.RandomRotation(30), # 随机旋转30度
    transforms.RandomResizedCrop(128), # 随机裁剪并调整为128X128
    # transforms.Resize((128, 128)),  # 将图形调整为128 X 128
                                transforms.ToTensor(),  # 将图形转换为张量
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])  # 标准化

# 加载图像
image = Image.open('../images/cat.jpg')

# 应用预处理
image_tensor = transform(image)
print(image_tensor.shape)  # 输出张量的形状
