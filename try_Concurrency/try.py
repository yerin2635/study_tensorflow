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
sleep_5()
sleep_10()
end_time = time.time()

print('It costs '+str(end_time - start_time)+' s')