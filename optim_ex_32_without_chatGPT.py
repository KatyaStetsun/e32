import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert


# Define the gradient G_k = A x_k - b
def gradient(A, B, X):
    return np.dot(A, X) - B

n=20

def gradient_descent_mieux(A, B, eps=1e-10, iter=5000):
    X = np.ones(n)*3
    for I in range(iter):
        grad = gradient(A, B, X)
        ro = np.dot(grad,grad)/np.dot(np.dot(A,grad),grad)
        X_new = X - ro * grad
        
        if np.linalg.norm(X_new - X) < eps:
            print(I)
            break
        
        X = X_new
    
    return X


A = hilbert(n)
B = np.dot(A, np.ones(n))

X_approx = gradient_descent_mieux(A, B)
X_theoretical = np.ones(n)

plt.figure(figsize=(10, 6))
plt.plot(range(n), X_approx, label="Gradient Descent (Optimal Step)")
plt.plot(range(n), X_theoretical, label="Theoretical Solution")

plt.show()

#graphick - it is what we need to have
#added this code to the github