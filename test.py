import torch
import torchvision.transforms as transforms
from PIL import Image

alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z']

# load the saved ResNet model
model = torch.load('model.pth')

# switch model to evaluation mode
model.eval()

# define the image transforms
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# load the test image
test_image = Image.open('test.jpg')

# apply the transforms to the test image
test_image_tensor = image_transforms(test_image)

# add batch dimension to the image tensor
test_image_tensor = test_image_tensor.unsqueeze(0)

# get the model's prediction
with torch.no_grad():
    prediction = model(test_image_tensor)

# get the predicted class index
predicted_class_index = torch.argmax(prediction).item()

# print the predicted class index
print(alpha[predicted_class_index])
