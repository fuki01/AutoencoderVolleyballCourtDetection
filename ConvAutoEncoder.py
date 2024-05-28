from torch.utils.data import Dataset
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        # inputsize(720*480*3)
        # Encoder
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2))
        self.t_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.t_conv3 = nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.t_conv4 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2))
        self.t_conv5 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.t_conv6 = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.t_conv7 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2))
        self.t_conv8 = nn.ConvTranspose2d(64, 1, kernel_size=(3, 3), stride=(2, 2), output_padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = F.relu(self.t_conv5(x))
        x = F.relu(self.t_conv6(x))
        x = F.relu(self.t_conv7(x))
        x = self.t_conv8(x)
        return x


# モデル定義など


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_filenames = os.listdir(os.path.join(root_dir, 'train_images'))
        self.label_filenames = os.listdir(os.path.join(root_dir, 'train_labels'))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'train_images', self.image_filenames[idx])
        label_path = os.path.join(self.root_dir, 'train_labels', self.label_filenames[idx])
        img = Image.open(img_path).convert('RGB')
        # 720*480*3
        img = img.resize((720, 480))
        # Tensor
        img = transforms.ToTensor()(img)
        label = Image.open(label_path).convert('L')
        # 720*480
        label = label.resize((720, 480))
        # 2値画像に変換
        label = transforms.ToTensor()(label)
        return img, label


# バッチサイズ、エポック数、学習率などの設定
batch_size = 8
num_epochs = 1000
learning_rate = 0.001

# DataLoaderを用意
# dataset = MyDataset(input_images, target_images)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


dataset = CustomDataset('C:\\Users\\sawano\\Desktop\\caeCourtDetection\\dataset')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルの定義
model = ConvAutoEncoder()

# GPUを使う場合は、以下を実行
# device = torch.device('cpu')
# model.to(device)

# 損失関数の定義
criterion = nn.MSELoss()

# 最適化手法の定義
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 学習のループ
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # モデルによる推論
        outputs = model(inputs)

        # 損失計算
        loss = criterion(outputs, targets)

        # 誤差逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()

        running_loss += loss.item()
        # モデルを１００回に１回保存
        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'C:\\Users\\sawano\\Desktop\\caeCourtDetection\\model\\'+str(epoch)+'_caeCourtDetection.pth')

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, running_loss/len(dataloader)))


# モデルの保存
torch.save(model.state_dict(), 'C:\\Users\\sawano\\Desktop\\caeCourtDetection\\model\\caeCourtDetection.pth')
