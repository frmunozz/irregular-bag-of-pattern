from tqdm.contrib.concurrent import process_map  # or thread_map
import time
import numpy as np


def _foo(my_number):
   square = my_number * my_number
   time.sleep(0.5)
   return square


def _foo2(arr):
    s = np.sum(arr)
    time.sleep(2)
    return s


if __name__ == '__main__':
    x = np.array([[1, 2, 3],
                  [5, 5, 5],
                  [1, 1, 1],
                  [9, 1, 2]])

    r = process_map(_foo2, x, max_workers=4)

    print(r)
