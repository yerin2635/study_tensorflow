import os
import threading
import time

def sleep_5():
    for i in range(0, 5):
        print(i)
        time.sleep(1)
    return

def sleep_10():
    for i in range(0, 10):
        print(i)
        time.sleep(1)
    return
  
start_time = time.time()

thread_1 = threading.Thread(target=sleep_5)  # 例項化一個執行緒物件，使執行緒執行這個函式
thread_2 = threading.Thread(target=sleep_10)  # 例項化一個執行緒物件，使執行緒執行這個函式
thread_1.start()  # 啟動這個執行緒
thread_2.start()  # 啟動這個執行緒
thread_1.join()  # 等待thread_1結束，如果不打join程式會直接往下執行
thread_2.join()  # 等待thread_2結束，如果不打join程式會直接往下執行

end_time = time.time()
print('It costs '+str(end_time - start_time)+' s')