import RK_Project
import numpy as np
import matplotlib.pyplot as plt
import math
def my_f(x):
    return 1 + np.exp(-1.0 * x)

def my_f_dot(t,x):
    return -1.0 * np.exp(-1.0 * t)



(x , true_step) = np.linspace(0,2.5,250, retstep=True)
int_step = true_step * 10
array_size = math.ceil(1 + (2.5 - 0)/ np.asscalar(int_step))
int_inputs = np.zeros(array_size)
for i in range(array_size):
    int_inputs[i] = i*int_step

int_soln = RK_Project.RK4(my_f_dot, 2, int_step, 1, [0,2.5])
int_ss = RK_Project.RK4(my_f_dot, 2, int_step, 2, [0,2.5], acceptable_error=0.00001)
print("RK Integration Steady State detected at y = ", int_ss)
plt.plot(x, my_f(x), label='ideal')
plt.scatter(int_inputs, int_soln, label='data', edgecolors='r', facecolors='none')
plt.show()