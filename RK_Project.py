
import numpy as np
import math


def RK4(f_dot, init_value, step_size, run_mode = 1, integration_range = [0,1], acceptable_error = 0.001, max_iter = 1000):

    if run_mode == 1:
        f_array = RK4_a_to_b(f_dot, init_value, step_size, integration_range)
    elif run_mode == 2:
        f_array = RK4_to_steady_state(f_dot, init_value, step_size, integration_range, acceptable_error, max_iter)

    return f_array

# scalar integration using Runge Kutta 4th order method
def RK4_a_to_b(f_dot, init_value, step_size, integration_range):

    # initialize the return array
    array_size = math.ceil(1 + (integration_range[1] - integration_range[0]) / np.asscalar(step_size))
    output_array = np.zeros(array_size)
    
    # initialize the integrating constants
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0

    i = 0
    output_array[i] = init_value
    f_input = integration_range[0]
    for i in range(array_size - 1):
        k1 = step_size * f_dot(f_input, output_array[i] )
        k2 = step_size * f_dot(f_input + step_size / 2.0, output_array[i] + k1 / 2.0)
        k3 = step_size * f_dot(f_input + step_size / 2.0, output_array[i] + k2 / 2.0)
        k4 = step_size * f_dot(f_input + step_size, output_array[i] + k3)
        output_array[i+1] = output_array[i] + ( k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        f_input += step_size

    return output_array


# scalar integration until steady state using Runge Kutta 4th order method
def RK4_to_steady_state(f_dot, init_value, step_size, integration_range, acceptable_error, max_iter):

    # initialize loop parameters
    error_between_steps = 100
    integrating = True

    # initialize the integrating constants
    k1 = 0
    k2 = 0
    k3 = 0
    k4 = 0

    last_steady_state = init_value # init "previous" with init_value because of loop structure
    steady_state = init_value + 100 # ensure the iteration kicks off
    f_input = integration_range[0]
    while integrating: 
        k1 = step_size * f_dot(f_input, last_steady_state) 
        k2 = step_size * f_dot(f_input + step_size / 2.0, last_steady_state + k1 / 2.0)
        k3 = step_size * f_dot(f_input + step_size / 2.0, last_steady_state + k2 / 2.0)
        k4 = step_size * f_dot(f_input + step_size, last_steady_state + k3)

        steady_state = last_steady_state  + ( k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0

        if (abs(steady_state - last_steady_state)) < acceptable_error:
            integrating = False
        else:
            f_input += step_size
            last_steady_state = steady_state

    return steady_state

	
# calculate and return the 4 integration constants for RK4
def RK4_calc_k(f_dot, f_input, f_i, step_size):
	k1 = step_size * f_dot(f_input, f_i)
	k2 = step_size * f_dot(f_input + step_size/2.0, f_i + k1/2.0)
	k3 = step_size * f_dot(f_input + step_size/2.0, f_i + k2/2.0)
	k4 = step_size * f_dot(f_input + step_size, f_i + k3)
	
	return np.array([k1, k2, k3, k4])
	
