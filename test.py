import sys
import threading
import time
import os


def test(n):
    for i in range(10):
        status[n] = 'x'*30
        print('\r' + ','.join(status).replace(',', '\n'))
        print('---------------------------------')
        time.sleep(1)


thread_num = 5  # 线程数
status = []
for i in range(thread_num):
    status.append('')
    locals()[f't_{i}'] = threading.Thread(target=test, args=(i, ))

for i in range(thread_num):
    locals()[f't_{i}'].start()

