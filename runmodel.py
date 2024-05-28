import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2


def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder,self).__init__()
        #inputsize(720*480*3)
        #Encoder
        self.conv1=nn.Conv2d(3,64,kernel_size=(3,3),stride=(2,2))
        self.conv2=nn.Conv2d(64,128,kernel_size=(3,3),stride=(2,2))
        self.conv3=nn.Conv2d(128,256,kernel_size=(3,3),stride=(2,2))
        self.conv4=nn.Conv2d(256,512,kernel_size=(3,3),stride=(2,2))
        
        #Decoder
        self.t_conv1=nn.ConvTranspose2d(512,256,kernel_size=(3,3),stride=(2,2))
        self.t_conv2=nn.ConvTranspose2d(256,256,kernel_size=(3,3),stride=(1,1),padding=1)
        self.t_conv3=nn.ConvTranspose2d(256,256,kernel_size=(3,3),stride=(1,1),padding=1)
        self.t_conv4=nn.ConvTranspose2d(256,128,kernel_size=(3,3),stride=(2,2))
        self.t_conv5=nn.ConvTranspose2d(128,128,kernel_size=(3,3),stride=(1,1),padding=1)
        self.t_conv6=nn.ConvTranspose2d(128,128,kernel_size=(3,3),stride=(1,1),padding=1)
        self.t_conv7=nn.ConvTranspose2d(128,64,kernel_size=(3,3),stride=(2,2))
        self.t_conv8=nn.ConvTranspose2d(64,1,kernel_size=(3,3),stride=(2,2),output_padding=1)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        
        x=F.relu(self.t_conv1(x))
        x=F.relu(self.t_conv2(x))
        x=F.relu(self.t_conv3(x))
        x=F.relu(self.t_conv4(x))
        x=F.relu(self.t_conv5(x))
        x=F.relu(self.t_conv6(x))
        x=F.relu(self.t_conv7(x))
        x=self.t_conv8(x)
        return x


model = ConvAutoEncoder()

cap = cv2.VideoCapture("C:\\Users\\sawano\\Downloads\\USA vs. Russia – Full Volleyball Match - Rio 2016 _ Throwback Thursday_play_sence.mp4")

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# 動画のフレーム数を取得
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 動画のFPSを取得
fps = cap.get(cv2.CAP_PROP_FPS)
# 動画のフレームサイズを取得
size = (720,480)

# 動画を保存するための設定
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
video = cv2.VideoWriter('C:\\Users\\sawano\\Desktop\\caeCourtDetection\\output\\out.avi', fourcc, 25, size)

for i in range(frame_count-1):
    print(i)
    # フレームを取得
    ret, frame = cap.read()
    if ret == False:
        break
    # 画像を保存する
    img = cv2pil(frame)

    # 720*480
    img = img.resize((720, 480))
    # Tensor
    img = transforms.ToTensor()(img)

    # GPUを使う場合は、以下を実行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    model.to(device)
    model.load_state_dict(torch.load('C:\\Users\\sawano\\Desktop\\caeCourtDetection\\model\\110_caeCourtDetection.pth'))


    # 画像をモデルに入力し、出力を得る
        # 画像をモデルに入力し、出力を得る

    output = model(img)

    # 画像を保存する
    output = output.to('cpu')
    output = output.detach().numpy()
    output = np.squeeze(output)
    output = output * 255
    output = output.astype(np.uint8)

    # outputを２値化する
    ret, output = cv2.threshold(output, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 収縮処理
    output = cv2.erode(output, np.ones((3, 3), np.uint8), iterations=5)
    output = cv2.dilate(output, np.ones((3, 3), np.uint8), iterations=10)


    # 領域が一番多きものを残して、それ以外を黒くする
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_index = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_index = cnt

    output = np.zeros(output.shape, np.uint8)
    cv2.drawContours(output, [max_index], 0, 255, -1)

    # 膨張処理
    output = cv2.dilate(output, np.ones((3, 3), np.uint8), iterations=1)

    video.write(output)
    frame = cv2.resize(frame, (720, 480))

    # frameとoutputを結合してひょうじする
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    output = cv2.addWeighted(frame, 0.5, output, 0.5, 0)





    cv2.imshow("output", output)
    cv2.waitKey(1)
    # cv2.imwrite("C:\\Users\\sawano\\Desktop\\caeCourtDetection\\output\\" + str(i) + 
                # ".png", output)

video.release()
cap.release()
