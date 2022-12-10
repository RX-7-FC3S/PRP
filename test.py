import sys
import threading
import time
import os


def test(n):
    for i in range(10):
        status[f'线程{n}'] = i
        print('\r' + str(status), end='')
        time.sleep(1)


thread_num = 5  # 线程数
status = {}
for i in range(thread_num):
    status[f'线程{i}'] = ''
    locals()[f't_{i}'] = threading.Thread(target=test, args=(i, ))
print(status)
for i in range(thread_num):
    locals()[f't_{i}'].start()

