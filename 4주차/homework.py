import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

cap = cv2.VideoCapture('videos/01.mp4')
sticker_img = cv2.imread('imgs/pig.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()

    if ret == False:
        break

    dets = detector(img)

    for det in dets:
        shape = predictor(img, det)  # 랜드마크(점) 저장

        try:
            x1 = det.left()#얼굴 좌측 
            y1 = det.top()#좌측 위
            x2 = det.right()
            y2 = det.bottom()

            center_x=shape.parts()[4].x#코 x
            center_y = shape.parts()[4].y-10 #코 y

            h, w, c = sticker_img.shape
            nose_w=int((x2-x1)/4)
            nose_h=int(h/w*nose_w)

            nose_x1=int(center_x-nose_w/2)
            nose_x2=nose_x1+nose_w

            nose_y1 = int(center_y - nose_h / 2)
            nose_y2 = nose_y1 + nose_h

            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(nose_w, nose_h))

            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            img[nose_y1:nose_y2, nose_x1:nose_x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[
                                                                                                               nose_y1:nose_y2,
                                                                                                               nose_x1:nose_x2]



            # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
        except:
            pass

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break