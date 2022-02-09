import cv2

img=cv2.imread('01.jpg')


#cv2.rectangle(img, pt1=(259, 89), pt2=(380, 348), color=(255, 0, 0), thickness=-1)
#ptr1은 왼쪽위, pt2는 오른쪽아래 thickness에 채우려면 음수를 넣으면 영역을 채우게 됨

cropped_img = img[89:348, 259:380]

cv2.imshow('cropped', cropped_img)
cv2.waitKey(0)
img_resized = cv2.resize(img, (512, 256))

cv2.imshow('resized', img_resized)
cv2.waitKey(0)













