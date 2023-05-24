import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

# CUDA 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.init()

data_transforms = transforms.Compose([
    transforms.Resize(227),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 데이터 셋
train_dataset = datasets.ImageFolder(root='trainImages', transform=data_transforms)
test_dataset = datasets.ImageFolder(root='testImages', transform=data_transforms)

# 학습, 테스트 데이터 분리
# train_size = int(0.8 * len(image_dataset))  # 전체 데이터 비율 80%
# test_size = len(image_dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32,
                                         shuffle=True, num_workers=0)

# 학습, 검증 데이터 분리
train_size = int(0.8 * len(train_dataset))  # 검증 데이터 비율 80%
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=32,
                                          shuffle=True, num_workers=0)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=32,
                                        shuffle=True, num_workers=0)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# 모델 생성 후 L2 정규화 적용
resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34')
resnet.fc = nn.Linear(512, 26)  # 출력층의 뉴런 수는 10

# L2 정규화 적용할 가중치 파라미터를 모아둘 리스트
weight_decay_params = []
bias_params = []

# 정규화 비율 설정
weight_decay = 0.001

# 모든 가중치 파라미터를 추출하여 정규화 적용
for name, param in resnet.named_parameters():
    if 'bias' in name:
        bias_params.append(param)
    else:
        weight_decay_params.append(param)

# 모델 학습을 위한 변수 설정
train_losses = []
val_losses = []
accuracies = []

# 모델 학습을 위한 하이퍼파라미터 설정
criterion = nn.CrossEntropyLoss()
# 정규화를 위한 optimizer 생성
optimizer = optim.SGD([
    {'params': weight_decay_params, 'weight_decay': weight_decay},
    {'params': bias_params, 'weight_decay': 0.0}
], lr=0.001, momentum=0.9)

# 모델 학습
resnet.to(device)
for epoch in range(15):
    running_loss = 0.0
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2400 == 2399:
            train_loss = running_loss / 100
            train_losses.append(train_loss)
            #print('[epoch :%d, %3d] loss: %.3f' % (epoch+1, i+1, running_loss/100))
            running_loss = 0.0

    # 검증 데이터셋을 이용하여 모델 성능 평가
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = resnet(images)
            val_loss = loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valloader)
    val_losses.append(val_loss)
    accuracy = 100 * val_correct / val_total
    accuracies.append(accuracy)
    print('[epoch :%d] train loss: %.3f, val loss: %.3f' % (epoch+1, train_losses[-1], val_losses[-1]))
    print('Accuracy of the network on the validation images: %d %%' % accuracy)

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

test_accuracy = 100 * correct / total
print('Accuracy of the network on the test images: %d %%' % test_accuracy)

# 모델 구조 저장
torch.save(resnet, 'resnet34.pth')

# 결과 저장
result_file = "result_resnet34.pth"
with open(result_file,"w") as f:
    for epoch in range(20):
        # 결과 파일에 기록
        f.write("Epoch: {}\n".format(epoch+1))
        f.write("Train Loss: {:.3f}\n".format(train_losses[-1]))
        f.write("Val Loss: {:.3f}\n".format(val_losses[-1]))
        f.write("Accuracy: {:.2f}%".format(accuracies[-1]))
        f.write("\n")
    f.write("Test Accuracy: {:.2f}%\n".format(test_accuracy))