import numpy as np
'''
# initialize variables
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
'''
def BNMtF(A,B,S,C,delta,max_iterations):
	# For loop
	i,e = [],[]
	for iteration in range(max_iterations):
		B_mp = np.divide((A @ C.T @ S.T), (B @ B.T @ A @ C.T @ S.T) + delta)
		B = np.multiply(B, B_mp)

		C_mp = np.divide(S.T @ B.T @ A, S.T @ B.T @ A @ C.T @ C + delta)
		C = np.multiply(C, C_mp)

		S_mp = np.divide(B.T @ A @ C.T, B.T @ B @ S @ C @ C.T + delta)
		S = np.multiply(S, S_mp)

		# print error
		error = np.linalg.norm(A - B@S@C ,'fro') ** 2
		if iteration % 10:
			print(error)
			i.append(iteration)
			e.append(error)
	return i,e

