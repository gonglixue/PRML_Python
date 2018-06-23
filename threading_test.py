import threading
from time import ctime
from time import sleep
import math

def append_list(target_list, i):
    target_list.append(i)
    for t in range(10000):
        temp = t ** 2
        temp = math.sqrt(temp)
    sleep(5)
    print("append %s. %s" % (i, ctime()))

my_list = []

threads = []



# t1 = threading.Thread(target=append_list, args=(my_list, 1))
# threads.append(t1)
# t2 = threading.Thread(target=append_list, args=(my_list, 3))
# threads.append(t2)

for i in range(100):
    t = threading.Thread(target=append_list, args=(my_list, i))
    threads.append(t)

for t in threads:
    t.setDaemon(True)
    t.start()

print("all over")
for t in threads:
    t.join()


for i in range(len(my_list)):
    print(my_list[i])