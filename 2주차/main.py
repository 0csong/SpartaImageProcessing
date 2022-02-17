import cv2
import numpy as np

net=cv2.dnn.readNetFromTorch('models/eccv16/la_muse.t7')



img=cv2.imread('imgs/02.jpg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))#이미지의 원본비율을 유지하면서 크기 변형
img=img[162:513,185:428]
print(img.shape)
MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)#blobFromImage는 전달인자이미지를 전처리한다
print(blob.shape)#blobFromImage는 차원을 조정해줌(딥러닝 모델에 맞게해주는것)

net.setInput(blob)
output=net.forward()

output = output.squeeze().transpose((1, 2, 0))#squeeze늘린차원 줄여줌,transposesms 차원변형을 거꾸로
output += MEAN_VALUE

output = np.clip(output, 0, 255)#clip은 제한
output = output.astype('uint8')#정수로 바꿔줘라


cv2.imshow('output',output)
cv2.imshow('img',img)
cv2.waitKey(0)

