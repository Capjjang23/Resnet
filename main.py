import os
import shutil

import zipfile

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

# 모든 이미지  클래스별 폴더로 저장
# data_dir = 'Images_227'  # 이미지 파일이 있는 폴더 경로
# output_dir = 'data'  # 복사할 폴더 경로
# for filename in os.listdir(data_dir):
#     if filename.endswith('.jpg'):  # 이미지 파일이면
# #         print(filename.split('_')[0])
#         class_name = filename.split('_')[0][0]  # 클래스 이름 추출
#         class_dir = os.path.join(output_dir, class_name)  # 클래스 폴더 경로
#         if not os.path.exists(class_dir):
#             os.makedirs(class_dir)  # 클래스 폴더 생성
#         src_path = os.path.join(data_dir, filename)  # 원본 파일 경로
#         dst_path = os.path.join(class_dir, filename)  # 이동할 파일 경로
#         shutil.copy(src_path, dst_path)  # 파일 복사
#
#
#
# 나눠진 파일 압축하기
# my_zip = zipfile.ZipFile('data.zip', 'w')
#
# for dirname, _, filenames in os.walk("data"):
#     for filename in filenames:
#         # load audio file with Librosa
#         file_path = dirname + '/' + filename
#         my_zip.write(file_path)
#
# my_zip.close()


# CUDA 초기화
from torchvision.models import ResNet18_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.init()

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 데이터 셋
image_dataset = datasets.ImageFolder(root='Images_227', transform=data_transforms)

# 학습, 테스트 데이터 분리
train_size = int(0.8 * len(image_dataset))  # 전체 데이터 비율 80%
test_size = len(image_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                         shuffle=False, num_workers=0)

# 학습, 검증 데이터 분리
train_size = int(0.8 * len(train_dataset))  # 검증 데이터 비율 80%
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=False, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                        shuffle=False, num_workers=0)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ResNet18 모델 생성
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', 'weights=ResNet18_Weights.IMAGENET1K_V1')
resnet.fc = nn.Linear(512, 26) # 출력층의 뉴런 수는 10

# 모델 학습을 위한 하이퍼파라미터 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

# 모델 학습
resnet.to(device)
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[epoch :%d, %3d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
            running_loss = 0.0

    # 검증 데이터셋을 이용하여 모델 성능 평가
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation images: %d %%' % (100 * correct / total))

# 모델 평가
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = resnet(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 모델 가중치와 state_dict 저장
#checkpoint = {'model_state_dict': resnet.state_dict()}
#torch.save(checkpoint, 'resnet18_model.pth')