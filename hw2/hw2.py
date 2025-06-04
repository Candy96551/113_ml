from random import random, randint, choice
import numpy as np


citys = [
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
]

def distance(p1, p2):
    #print('p1=', p1)
    x1, y1 = p1
    x2, y2 = p2
    return ((x2-x1)**2+(y2-y1)**2)**0.5

def pathLength(c):
    dist = 0
    plen = len(c)
    for i in range(plen):
        dist += distance(c[i],c[(i+1)%plen])
    return dist

def randCity() :
    return randint(0, len(citys)-1)

def neighbor(c):    # 單變數解答的鄰居函數。
        fills = c.copy()
        i = randCity()
        j = randCity()
        t = fills[i]
        fills[i] = fills[j]
        fills[j] = t
        return fills      # 建立新解答並傳回。


def hillClimbing(x, pathLength, neighbor, max_fail):
    print("start: ", pathLength(x))             # 印出初始解
    fail = 0
    gens = 0
    while True:
        nx = neighbor(x)
        #print('.',nx)
        if pathLength(nx) < pathLength(x):
            x = nx
            gens += 1
            print(gens,  ':', pathLength(x), x)
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                print("solution: ", pathLength(x))
                return x

# 執行爬山演算法
hillClimbing(citys, pathLength, neighbor, max_fail=10000)
