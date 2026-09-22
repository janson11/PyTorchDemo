import torch
import torchvision

# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)
model.eval()

# 示例输入
example_input = torch.rand(1, 3, 224, 224)

# 方法1：通过追踪(tracing)导出
traced_script = torch.jit.trace(model, example_input)
# traced_script.save("resnet18_traced.pt")
traced_script.save("resnet18.pt")

# 方法2：通过脚本(scripting)导出
# scripted_model = torch.jit.script(model)
# scripted_model.save("resnet18_srcipted.pt")