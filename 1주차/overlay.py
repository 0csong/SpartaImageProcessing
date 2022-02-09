import cv2

img = cv2.imread('01.jpg')
overlay_img = cv2.imread('dices.png', cv2.IMREAD_UNCHANGED)

overlay_img = cv2.resize(overlay_img, dsize=(150, 150))
overlay_alpha = overlay_img[:, :, 3:] / 255.0#[높이:, 너비:,채널 3:] 그중 채널은 BGRA이므로 마지막 투명도인 A만 사용하기 위해 3으로 자른다, overlay_alpha는 투명한 정도를 나타내는 이미지
background_alpha = 1.0 - overlay_alpha #배경 투명도
#불투명한 부분은 0이 아닌 값, 투명한 부분(어두운 부분)은 0이다
# abc=overlay_alpha * overlay_img[:, :, :3]#overlay_img는 투명한 부분은 색값이 0이되게함 불투명한 부분은 색값이 0이상의 값
# cv2.imshow('dl',abc)
# cv2.waitKey(0)
x1 = 100
y1 = 100
#x1,y1은 주사위 이미지를 넣고싶은 왼쪽 위좌표
x2 = x1 + 150
y2 = y1 + 150
#x2,y2는 주사위 이미지 오른쪽아래
# img[y1:y2, x1:x2]=background_alpha * img[y1:y2, x1:x2]
# cv2.imshow('img',img)
# cv2.waitKey(0)
# img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3]
# cv2.imshow('img12',img)
# cv2.waitKey(0)
img[y1:y2, x1:x2] = overlay_alpha * overlay_img[:, :, :3] + background_alpha * img[y1:y2, x1:x2]#overlay_img[:, :, :3]은 앞에서 3개 즉 색정보
cv2.imshow('img',img)
cv2.waitKey(0)











