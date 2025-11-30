import numpy as np
from scipy import optimize

def test_fmin():
    true_x=3
    true_y=2
    def func(x):
        return (x - true_x) ** 2 + true_y
    
    initial_guess = 0
    result = optimize.fmin(func, initial_guess)
    print(f"True x: {true_x}")
    print(f"Optimized x: {result[0]}")
    print(f"Function value at true x: {func(true_x)}")
    print(f"Function value at optimized x: {func(result[0])}")
def test_minimize():
    def rosen(x):
        """The Rosenbrock function"""
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
    def rosen_der(x):
        """The derivative (i.e. gradient) of the Rosenbrock function
        """
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
        der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2] ** 2)
        return der
    
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    result=optimize.fmin(rosen, x0,  disp=True)
    print("Optimized parameters (fmin):", result)
    print("Function value at optimized parameters (fmin):", rosen(result))
    print("-----"*15)
    res = optimize.minimize(rosen, x0, method='BFGS', jac=rosen_der,
               options={'gtol': 1e-6, 'disp': True})
    print("Optimized parameters:", res.x)
    print("Function value at optimized parameters:", res.fun)
        
    
if __name__ == "__main__":
    test_fmin()
    print("===="*15)
    test_minimize()