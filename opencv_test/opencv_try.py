import cv2
import numpy as np
from matplotlib import pyplot as plt


















def read():
    img = cv2.imread('opencv_test/test.png')
    return img


def read_Change_picture_grayscale():
    img_gray1 = cv2.cvtColor(read(), cv2.COLOR_BGR2GRAY)  # 轉換為灰階圖片
    return img_gray1


def read_and_show(img):
    cv2.imshow('Image', img)
    wait_and_close()


def wait_and_close():
    cv2.waitKey(0)  # 等待關閉視窗的時間 0為持續等待直到按下任何鍵
    cv2.destroyAllWindows()  # 關閉所有opencv的視窗


def write(img):
    cv2.imwrite('opencv_test/write_test.jpg', img)  # 寫入圖片


def turn_over():
    img_flip = cv2.flip(read(), 1)
    read_and_show(img_flip)

# 取得圖片中所填的像素的範圍


def Cut_picture():

    ltop = (100, 100)
    rtbm = (200, 200)
    img_cap = read_Change_picture_grayscale()[ltop[1]:rtbm[1], ltop[0]:rtbm[0]]
    read_and_show(img_cap)

# 啟用相機


def camera():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('video_test.mp4')
    while (True):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def edge_detection():
    img = read_Change_picture_grayscale()
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 150)
    read_and_show(read())
    read_and_show(img)
    read_and_show(canny)


def threshold():
    plt.rcParams['font.family'] = ['Taipei Sans TC Beta']

    img = cv2.imread('opencv_test/test.png', 0)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    titles = ['初始圖片', '黑白轉換', '白黑轉換', '截斷閥值化處理', '低閥值零處理', '超閥值零處理']
    images = [img, th1, th2, th3, th4, th5]

    for i in range(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def blur_filtering():
    plt.rcParams['font.family'] = ['Taipei Sans TC Beta']
    img = cv2.imread('opencv_test/test.png')
    blur = cv2.blur(img, (5, 5))

    plt.subplot(121),plt.imshow(img) ,plt.title('圖片')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(blur),plt.title('Blur')
    plt.xticks([]), plt.yticks([])
    plt.show()

def gauss_filtering():
    plt.rcParams['font.family'] = ['Taipei Sans TC Beta']
    img = cv2.imread('opencv_test/test.png')
    gauss = cv2.GaussianBlur((img),(5,5),0)

    plt.subplot(121),plt.imshow(img) ,plt.title('圖片')
    plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(gauss),plt.title('GaussianBlur')
    plt.xticks([]), plt.yticks([])
    plt.show()

def draw_circle():
    image  = np.zeros((480,640,3),np.uint8) #uint8 無符號整數(0-255)
    image[:]=(128,128,128)
    # img.fill(128)
    color = (255,255,0)

    cv2.circle(image,(300,200),50,color,1)
    cv2.imshow('result',image)
    cv2.waitKey(0)





if __name__ == '__main__':
    threshold()
