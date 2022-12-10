import sys
import threading
import time
import os


def test():
    x = 0
    for i in range(3):
        sys.stdout.flush()
        sys.stdout.write('\n--->' + str(i))
        # print('--->' + str(i), end='\n')
        x = x + 1
        time.sleep(1)
    return x


thread_num = 5  # 线程数
for i in range(thread_num):
    locals()[f't_{i}'] = threading.Thread(target=test)
    locals()[f't_{i}'].start()
