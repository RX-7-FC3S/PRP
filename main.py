import sqlite3
import time

from compare import main
import eventlet
import threading


if __name__ == '__main__':

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
        print(1)
        db = sqlite3.connect('sqlite.db')
        cursor = db.cursor()
        # try:
        #     cursor.execute('DROP TABLE RESULT')
        # except Exception as e:
        #     print(e)
        # cursor.execute('''CREATE TABLE RESULT (id INTEGER PRIMARY KEY AUTOINCREMENT)''')

        fields = []
        for i in range(min_num, max_num + 1):
            field_name = 'NUM_{}'.format(i)
            fields.append(field_name)
            try:
                cursor.execute('ALTER TABLE RESULT ADD {} FLOAT'.format(field_name))
            except sqlite3.OperationalError as e:
                pass

        for i in range(0, num_scene):
            res = []
            time_start = time.perf_counter()
            for j in range(min_num, max_num + 1):
                rate = 1
                eventlet.monkey_patch()
                with eventlet.Timeout(time_limitation, False):
                    rate = main(rows, cols, shf_size, start, j)
                res.append(str(round(rate)))
                dur = time.perf_counter() - time_start
                print('\r线程'+ str(n) + ';情景:' + str(i), '; 拣货数量:' + str(j), '; 情景耗时:' + str(round(dur)) + 's', end='')
            cursor.execute('INSERT INTO RESULT ({}) VALUES ({})'.format(','.join(fields), ','.join(res)))
            db.commit()

    # 多线程
    thread_num = 10  # 线程数
    for i in range(thread_num):
        locals()[f't_{i}'] = threading.Thread(target=clac, args=(i,))
        locals()[f't_{i}'].start()
