from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')#얼굴탐지
model = load_model('models/mask_detector.model')#마스크탐지

# cap = cv2.VideoCapture('videos/04.mp4')
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape
    # 이미지 전처리하기
    blob = cv2.dnn.blobFromImage(img, size=(300, 300), mean=(104., 177., 123.))#300,300으로 resize ,mean값으로 빼준후 차원변형

    # 얼굴 영역 탐지 모델로 추론하기
    facenet.setInput(blob)
    dets = facenet.forward()#얼굴영역 탐지결과 저장

    # 각 얼굴에 대해서 반복문 돌기
    for i in range(dets.shape[2]):#dets.shape[2]는 얼굴개수
        confidence = dets[0, 0, i, 2]#해당얼굴의 신뢰도

        if confidence < 0.5:#50%이하는 무시
            continue

        # 사각형 꼭지점 찾기
        x1 = int(dets[0, 0, i, 3] * w)#얼굴위치가 전체이미지의 %로 나오게 되어서 원래 이미지 높이, 너비 곱해줌
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        face = img[y1:y2, x1:x2]#얼굴 영역을 잘라냄

        face_input = cv2.resize(face, dsize=(224, 224)) #face를 resize
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)#BGR에서 RGB로 왜나면 학습이 그렇게 된 모델임
        face_input = preprocess_input(face_input)#전처리 함수
        face_input = np.expand_dims(face_input, axis=0)#224,224,3->1,224,224,3

        mask,nomask=model.predict(face_input).squeeze()#차원 축소

        if mask>nomask:
            color=(0,255,0)
            label = 'Mask %d%%' % (mask * 100)
        else:
            color=(0,0,255)
            label = 'No Mask %d%%' % (nomask * 100)
        # 사각형 그리기
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color)
        cv2.putText(img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color,
                    thickness=2)

    cv2.imshow('result', cv2.resize(img,(1000,500)))
    if cv2.waitKey(1) == ord('q'):
        break
