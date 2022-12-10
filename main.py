import sqlite3
import time
from compare import main
import eventlet
import threading


def battle():

    # 定义算例参数
    rows = 14
    cols = 13
    shf_size = 5
    start = 6

    # 定义计算参数
    min_num = 1
    max_num = 15
    num_scene = 20
    time_limitation = 30

    def clac(n):
        print(f'线程{n}启动！')
        db = sqlite3.connect('sqlite.db')
        cursor = db.cursor()
        # try:
        #     cursor.execute(f'DROP TABLE RESULT{opponent}')
        # except Exception as e:
        #     print(e)
        #
        # cursor.execute(f'CREATE TABLE RESULT{opponent} (id INTEGER PRIMARY KEY AUTOINCREMENT)')

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
                res.append(str(round(rate)))
                dur = time.perf_counter() - time_start
                print('线程'+ str(n) + ';情景:' + str(i), '; 拣货数量:' + str(j), '; 情景耗时:' + str(round(dur)) + 's', end='')
            cursor.execute(f'INSERT INTO RESULT{opponent} ({",".join(fields)}) VALUES ({",".join(res)})')
            db.commit()

    # 多线程
    for i in range(threads):
        locals()[f't_{i}'] = threading.Thread(target=clac, args=(i,))
        locals()[f't_{i}'].start()


if __name__ == '__main__':
    # 选择比较对象: 0 代表与"按顺序拣货对比", 1 代表与"最近点拣货对比"
    opponent = 1
    # 线程数
    threads = 50

    battle()
