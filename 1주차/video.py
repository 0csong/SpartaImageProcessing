import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    if ret == False:
        break
    cv2.rectangle(img, pt1=(100, 183), pt2=(300, 465), color=(255, 255, 255), thickness=2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#흑백영상
    img = cv2.resize(img, dsize=(640, 360))#영상 resize
    # img = img[100:200, 150:250]#영상 일부분 자르기
    cv2.imshow('result', img)

    if cv2.waitKey(10) == ord('q'):
        break