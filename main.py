import sqlite3
import time
from compare import main
import eventlet
import threading
import os


# 定义算例参数
all_time_start = time.perf_counter()

rows = 14
cols = 13
shf_size = 5
start = 6

# 定义计算参数
min_num = 1
max_num = 5
num_scene = 3
time_limitation = 30

# 选择比较对象: 0 代表与"按顺序拣货对比", 1 代表与"最近点拣货对比"
opponent = 1
# 线程数
threads_num = 2

status = ['Status: Inactive']


def print_status():
    while status[0] == 'Status: Inactive':
        os.system('clear')
        print(status[0])
        print('Waiting for GLPK')
        time.sleep(0.3)
        continue
    while status[0] == 'Status: Running':
        os.system('clear')
        print('\r', end='')
        print('\n'.join(status), end='')
        time.sleep(0.1)
    else:
        os.system('clear')
        print('\r', end='')
        print('\n'.join(status))
        print(f'本次总共计算{num_scene * threads_num}种情景\n累计耗时{round(all_dur)}s')
        # 退出主程序
        # Unix
        pid = os.system("kill $(ps -ef | grep python | grep -v grep | awk '{print $2}')")
        os.system(f'kill -9 {pid}')
        # Windows


def clac(n):
    print(f'\n线程{n}启动!')
    db = sqlite3.connect('sqlite.db')
    cursor = db.cursor()

    fields = []
    for i in range(min_num, max_num + 1):
        field_name = 'NUM_{}'.format(i)
        fields.append(field_name)
        try:
            cursor.execute(f'ALTER TABLE RESULT{opponent} ADD {field_name} FLOAT')
        except sqlite3.OperationalError as e:
            pass

    for i in range(0, num_scene):
        res = []
        time_start = time.perf_counter()
        for j in range(min_num, max_num + 1):
            rate = 1
            eventlet.monkey_patch()
            with eventlet.Timeout(time_limitation, False):
                rate = main(rows, cols, shf_size, start, j, opponent)
                status[0] = 'Status: Running'
            res.append(str(round(rate)))

            dur = time.perf_counter() - time_start
            status[n + 1] = f'线程:{n}; 情景:{i}; 拣货数量:{j}; 用时:{round(dur)}s;'

        cursor.execute(f'INSERT INTO RESULT{opponent} ({",".join(fields)}) VALUES ({",".join(res)})')
        db.commit()


print_thread = threading.Thread(target=print_status)
print_thread.start()

# 多线程
threads_list = []
for i in range(threads_num):
    status.append('')
    t = threading.Thread(target=clac, args=(i,))
    t.start()
    threads_list.append(t)

for thread in threads_list:
    thread.join()

all_dur = time.perf_counter() - all_time_start
status[0] = 'Status: Done'



