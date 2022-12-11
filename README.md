# Packing Routes Programming

## SQL in python
1. using sqlite3 to connect the local server sqlite database
2. using `DROP TABLE <TABLE NAME>` to delete a table
3. using `ALTER TABLE <TABLE NAME> ADD <FIELD NAME> <DATA TYPE>` to add a new field

## eventlet module
#### _to limit the running time of a function_

```python
import time
import eventlet


def fun():
    time.sleep(5)
    return 1


if __name__ == '__main__':
    eventlet.monkey_patch()
    flag = 0
    with eventlet.Timeout(3, False):  
    # 3 meas the time limitation is 3 seconds 
    # False means do not raise Exception
        flag = fun()
        
    print(flag)

# the print result is 0, because the fun() will be killed in 3 seconds 
```


## print()
#### _special usage_

```python
import time
import sys

# a default print()
print('value', sep=' ', end='\n', file=sys.stdout, flush=False)

# a progress bar by print
for i in range(100):
    print('\r', end='') # \r means remove the cursor to the head of line
    print(f'Progress:{i + 1}', '#' * (i + 1), end='')
    sys.stdout.flush()
    time.sleep(0.1)
```

## threading module
#### _multi threads_

```python
# create multi threads 
import threading


def fun(param):
    return param

threads_num = int  # the total number of threads you want to create

threads_list = []
for i in range(threads_num):
    t = threading.Thread(target=fun, args=('param', ))
    t.start()
    threads_list.append(t)

for thread in threads_list:
    thread.join() # to achieve that continuing to main thread until all sub-threads are done
```

