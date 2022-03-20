import cv2

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel('models/EDSR_x3.pb')#해상도 향상모델
sr.setModel('edsr', 3)#모델을 거치고나면 이미지크기가 3배늘어남

img=cv2.imread('imgs/06.jpg')

result=sr.upsample(img)

resized_img=cv2.resize(img,dsize=None,fx=3,fy=3)#xy로 3배씩

cv2.imshow('img',img)
cv2.imshow('result',result)#해상도향상거친 사진
cv2.imshow('resized_img',resized_img)#단순 3배
cv2.waitKey(0)