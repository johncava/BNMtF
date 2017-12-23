import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from BNMtF import *
from convBNMtF import *

m = 30
p = 10
o = 10
n = 20
delta = 1e-8
max_iterations = 1000
A = np.random.rand(m,p) @ np.random.rand(p, o) @ np.random.rand(o ,n)

# initialize B, S, C
B = np.random.rand(m,p)
S = np.random.rand(p,o)
C = np.random.rand(o,n)

i1, e1 = BNMtF(A,B,S,C,delta,max_iterations)
i2, e2 = convBNMtF(A,B,S,C,delta,max_iterations)
plt.plot(i1,e1)
plt.plot(i2,e2)
plt.legend(['BNMtF', 'convBNMtF'], loc='upper left')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
plt.savefig('results.png')
