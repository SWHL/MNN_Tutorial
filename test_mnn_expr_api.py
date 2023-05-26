import MNN.expr as F
from torchvision import transforms
from PIL import Image

mnn_model_path = './mobilenet_v2-b0353104.mnn'
vars = F.load_as_dict(mnn_model_path)
inputVar = vars["input"]
print('input shape: ', inputVar.shape)

image_path = './test.jpg'
input_image = Image.open(image_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)

# 推理模型
inputVar.write(input_tensor.tolist())

# 查看输出结果
outputVar = vars['output']
print('output shape: ', outputVar.shape)

cls_id = F.argmax(outputVar, axis=1).read()
cls_probs = F.softmax(outputVar, axis=1).read()

print("cls id: ", cls_id)
print("cls prob: ", cls_probs[0, cls_id])
