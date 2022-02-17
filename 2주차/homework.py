# #숙제 1. 액자 부분만 crop해서 추론하기
# import cv2
# import numpy as np
#
# net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
# img = cv2.imread('imgs/hw.jpg')
#
# h, w, c = img.shape
#
# img = cv2.resize(img, dsize=(800, int(h / w * 800)))
# img = img[93:229, 303:504]
#
# MEAN_VALUE = [103.939, 116.779, 123.680]
# blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
#
# net.setInput(blob)
# output = net.forward()
#
# output = output.squeeze().transpose((1, 2, 0))
#
# output += MEAN_VALUE
# output = np.clip(output, 0, 255)
# output = output.astype('uint8')
#
# cv2.imshow('img', img)
# cv2.imshow('output', output)
#
# cv2.waitKey(0)
#
# #2. 바뀐 이미지를 액자 안에 다시 집어넣기
# import cv2
# import numpy as np
#
# net = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')
#
# img = cv2.imread('imgs/hw.jpg')
#
# cropped_img = img[140:370, 480:810]
# cv2.imshow('cro',cropped_img)
#
# h, w, c = cropped_img.shape
#
# cropped_img = cv2.resize(cropped_img, dsize=(500, int(h / w * 500)))
#
# print(img.shape)
#
# MEAN_VALUE = [103.939, 116.779, 123.680]
# blob = cv2.dnn.blobFromImage(cropped_img, mean=MEAN_VALUE)
#
# print(blob.shape)
#
# net.setInput(blob)
# output = net.forward()
#
# output = output.squeeze().transpose((1, 2, 0))
# output = output + MEAN_VALUE
#
# output = np.clip(output, 0, 255)
# output = output.astype('uint8')
#
# output = cv2.resize(output, (w, h))
#
# img[140:370, 480:810] = output
#
# cv2.imshow('output', output)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# # 3. 영상에서 해보기
# import cv2
# import numpy as np
#
# net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
#
# cap = cv2.VideoCapture('imgs/03.mp4')
#
# while True:
#     ret, img = cap.read()
#
#     if ret == False:
#         break
#
#     MEAN_VALUE = [103.939, 116.779, 123.680]
#     blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)
#
#     net.setInput(blob)
#     output = net.forward()
#
#     output = output.squeeze().transpose((1, 2, 0))
#
#     output += MEAN_VALUE
#     output = np.clip(output, 0, 255)
#     output = output.astype('uint8')
#
#     cv2.imshow('img', img)
#     cv2.imshow('result', output)
#     if cv2.waitKey(1) == ord('q'):
#         break

#4. 가로로 3개 나누기
import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('models/instance_norm/mosaic.t7')
net2 = cv2.dnn.readNetFromTorch('models/instance_norm/the_scream.t7')
net3 = cv2.dnn.readNetFromTorch('models/instance_norm/candy.t7')

cap = cv2.VideoCapture('imgs/03.mp4')

while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h / w * 500)))

    MEAN_VALUE = [103.939, 116.779, 123.680]
    blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

    net.setInput(blob)
    output = net.forward()

    output = output.squeeze().transpose((1, 2, 0))

    output += MEAN_VALUE
    output = np.clip(output, 0, 255)
    output = output.astype('uint8')

    net2.setInput(blob)
    output2 = net2.forward()

    output2 = output2.squeeze().transpose((1, 2, 0))

    output2 += MEAN_VALUE
    output2 = np.clip(output2, 0, 255)
    output2 = output2.astype('uint8')

    net3.setInput(blob)
    output3 = net3.forward()

    output3 = output3.squeeze().transpose((1, 2, 0))

    output3 += MEAN_VALUE
    output3 = np.clip(output3, 0, 255)
    output3 = output3.astype('uint8')

    output = output[0:100, :]
    output2 = output2[100:200, :]
    output3 = output3[200:, :]

    output4 = np.concatenate([output, output2, output3], axis=0)

    cv2.imshow('result', output4)

    if cv2.waitKey(1) == ord('q'):
        break