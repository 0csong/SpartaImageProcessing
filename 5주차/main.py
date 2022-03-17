#그레이 스케일 사진에 색입히기(Lab 색파일)
import cv2
import numpy as np

proto = 'models/colorization_deploy_v2.prototxt'
weights = 'models/colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weights)

pts_in_hull = np.load('models/pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]

net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

img = cv2.imread('imgs/02.jpg')

h, w, c = img.shape

img_input = img.copy()

img_input = img_input.astype('float32') / 255.#딥러닝을 위해 소수로 바꾸기
img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2Lab)
img_l = img_lab[:, :, 0:1]

blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

net.setInput(blob)#이모델은 L채널을 받아 a,b채널을 예측하는모델
output=net.forward()

output=output.squeeze().transpose((1,2,0))

output_resize=cv2.resize(output,(w,h))
output_lab=np.concatenate([img_l,output_resize],axis=2)#0 가로 1세로 2채널방향

output_bgr=cv2.cvtColor(output_lab,cv2.COLOR_Lab2BGR)
output_bgr=output_bgr*255
output_bgr=np.clip(output_bgr,0,255)#0에서 255까지로 자름
output_bgr=output_bgr.astype('uint8')

cv2.imshow('img',img)
cv2.imshow('output',output_bgr)
cv2.waitKey(0)

















