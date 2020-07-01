import numpy as np

x = np.array([[-9,-5,5],
[4,-7,-11],
[7,6,-1],
[-9,-5,4],
[-5,-6,-1],
[-4,-4,-8],
[5,7,-9],
[2,-4,3],
[-6,1,7],
[-10,6,-7]])

w = np.array([-0.1, -0.3, 0.2])
b = 2
learning_rate = 0.02
desire = np.array([0,0,1,1,1,0,1,0,0,1])

y = np.matmul(x, w.T) + b
dE_dy = (y - desire) * 0.1
dE_dW = np.matmul(x.T,dE_dy)
dE_db = np.matmul(np.ones(y.shape[0]), y) * 0.1
w = w - (learning_rate *  dE_dW)
b = b - (learning_rate *  dE_db)

print(w)
print(b)