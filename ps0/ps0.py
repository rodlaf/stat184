import numpy as np
import matplotlib.pyplot as plt
import math

A = np.array([[0, 2, 4],
              [2, 4, 2],
              [3, 3, 1]])

b = np.array([[-2, -2, -4]]).T

c = np.array([[1, 1, 1]]).T



# 6.1.a
A_inv = np.linalg.inv(A)
print('A^-1 is\n', A_inv)

# 6.1.b
print('\nA^-1 b is\n', np.matmul(A_inv, b))
print('\nA c is\n', np.matmul(A, c))

# 6.2.a
n = 1000
Z = np.random.randn(n)

def F_hat(x):
    summand = 0
    for i in range(n):
        if Z[i] <= x:
            summand +=1
    return summand / n

vF_hat = np.vectorize(F_hat)
X = np.linspace(-3, 3, n)
plt.plot(X, vF_hat(X), label='Gaussian eCDF')
plt.xlabel('Observations')
plt.ylabel('Probability')
plt.legend()
plt.savefig('ps0_6_2_a')

# 6.2.b
k_values = [1, 8, 64, 512]

for k in k_values:
    Y_k = np.sum(np.sign(np.random.randn(n, k)) * np.sqrt(1./k), axis=1)
    def F_hat(x):
        summand = 0
        for i in range(n):
            if Y_k[i] <= x:
                summand +=1
        return summand / n
    vF_hat = np.vectorize(F_hat)
    plt.plot(X, vF_hat(X), label='k = ' + str(k))

plt.legend()
plt.savefig('ps0_6_2_b')
