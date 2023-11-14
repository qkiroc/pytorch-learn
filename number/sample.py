from PIL import Image
import torch
from torchvision import transforms
from MnistNet import Net

def main():
  # 步骤1: 准备图片
  image_path = "test.png"
  # 读取图片，并转换为灰度图（这样才能是1维的）
  image = Image.open(image_path).convert("L")

  # 步骤2: 调整图片大小
  desired_size = (28, 28)
  resized_image = image.resize(desired_size)

  # 步骤3: 加载模型
  model = Net()
  model.load_state_dict(torch.load("mnist_net.pth"))
  model.eval()

  # 步骤4: 图像预处理
  transform = transforms.Compose([transforms.ToTensor()])
  input_image = transform(resized_image).view(-1, 28*28)

  # 步骤5: 进行预测
  with torch.no_grad():
      output = model(input_image)
      predicted_class = torch.argmax(output, dim=1).item()
  print("Predicted class:", predicted_class)

if __name__ == "__main__":
  main()