import numpy as np

'''
# initialize variables
m = 30
p = 10
o = 10
n = 20
delta = 1e-8
step = 2
sigma = 1
alpha = 0.01
beta = 0.01
max_iterations = 1000

A = np.random.rand(m,p) @ np.random.rand(p, o) @ np.random.rand(o ,n)

# initialize B, S, C
B = np.random.rand(m,p)
S = np.random.rand(p,o)
C = np.random.rand(o,n)
'''
def J_B(A,B,S,C):
	return B @ S @ C @ C.T @ S.T - A @ C.T @ S.T + beta*B@B.T@B - beta*B

def J_C(A,B,S,C):
	return S.T @ B.T @ B @ S @ C - S.T @ B.T @ A + alpha* C @ C.T@ C - alpha*C

def J_S(A,B,S,C):
	return B.T @ B @ S @ C @ C.T - B.T @ A @C.T

def b_bar(b_ref,jb):
	b = np.copy(b_ref)
	for x in range(0,b.shape[0]):
		for y in range(0,b.shape[1]):
			if jb[x,y] < 0:
				b[x,y] = max(b[x,y], sigma)
	return b

def c_bar(c_ref,jc):
	c = np.copy(c_ref)
	for x in range(0,c.shape[0]):
		for y in range(0,c.shape[1]):
                        if jc[x,y] < 0:
                                c[x,y] = max(c[x,y], sigma)
	return c

def s_bar(s_ref,js):
	s = np.copy(s_ref)
	for x in range(0,s.shape[0]):
		for y in range(0,s.shape[1]):
                        if js[x,y] < 0:
                                s[x,y] = max(s[x,y], sigma)
	return s

def check(oldJ, newJ):
	#print(oldJ, newJ)
	if np.linalg.norm(newJ) <= np.linalg.norm(oldJ):
		return True
	else:
		return False

# For loop
def convBNMtF(A,B,S,C,delta,max_iterations):
	step = 2
	sigma = 1
	alpha = 0.01
	beta = 0.01
	for iteration in range(max_iterations):
		delta_B = delta
		while True:
			jb = J_B(A,B,S,C)
			b = b_bar(B, jb)
			B = B - np.divide(np.multiply(b,jb), B@S@C@C.T@S.T + beta*b@b.T@b + delta_B)
			delta_B = delta_B * step
			new_jb = J_B(A,B,S,C)
			if check(jb, new_jb):
				break
		delta_C = delta
		while True:
			jc = J_C(A,B,S,C)
			c = c_bar(C, jc)
			C = C - np.divide(np.multiply(c, jc), S.T@B.T@B@S@c + alpha*c@c.T@c + delta_C)
			delta_C = delta_C * step
			new_jc = J_C(A,B,S,C)
			if check(jc, new_jc):
				break
		delta_S = delta
		while True:
			js = J_S(A,B,S,C)
			s = s_bar(S, js)
			S = S - np.divide(np.multiply(s,js), B.T@B@s@C@C.T + delta_S)
			delta_S = delta_S * step
			new_js = J_S(A,B,S,C)
			if check(js, new_js):
				break
		# Print Error
		error = np.linalg.norm(A - B@S@C, 'fro') ** 2
		if iteration % 10 == 0:
			print(error)
