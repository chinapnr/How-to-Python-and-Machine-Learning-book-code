from time import time
from concurrent.futures import *
from memory_profiler import profile


def my_cal(a):
    j = 0
    for i in range(a):
        j = j + i
    print(j)
    return j


@profile
def run():
    list_01 = [1000000, 2000000, 1500000, 2500000, 3000000]
    start = time()
    pool = ProcessPoolExecutor(max_workers=10)
    list_02 = list(pool.map(my_cal, list_01))
    print(list_02)
    end = time()
    print('cost time {:f} s'.format(end - start))

if __name__ == '__main__':
    run()
