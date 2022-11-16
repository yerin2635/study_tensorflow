import cv2
import numpy as np

# img = cv2.imread('opencv_test/test.png')
# img_gray1 = cv2.imread('opencv_test/hi.jpg' , cv2.IMREAD_GRAYSCALE) # 轉換為灰階圖片

# cv2.imwrite('opencv_test/write_test.jpg', img) # 寫入圖片

# img_flip = cv2.flip(img, 1)  # 翻轉圖片

# cv2.imshow('Image', img) 

# 取得圖片中所填的像素的範圍
# ltop = (100, 100)
# rtbm = (200, 200)
# img_cap = img[ltop[1]:rtbm[1],ltop[0]:rtbm[0]]
# cv2.imshow('test',img_cap)

# 啟用相機 並存存一張照片
cap = cv2.VideoCapture(0)
# ret,frame = cap.read()
# cv2.imshow('frame', frame)

# cap.release() #釋放照相機

while(True):
    ret, frame = cap.read()
 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()

# cv2.waitKey(0)  # 等待關閉視窗的時間 0為持續等待直到按下任何鍵
# cv2.destroyAllWindows()  # 關閉所有opencv的視窗


