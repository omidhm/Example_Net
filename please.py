import numpy as np
import time
a = 1000000000

def fo(a):
    j = 0
    t1 = time.time()
    for i in range(a):
        j += 1
    
    print((time.time()-t1)/60)

def wh(a):
    t1 = time.time()
    while i<a:
        i += 1
        j += 1
    print((time.time()-t1)/60)

wh(a)
fo(a)