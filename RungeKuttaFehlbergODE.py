# Collin Stratton
# CST-305
# Topic 2 Project 2: Runge-Kutta-Fehlberg (RKF) for ODE
# Dr. Ricardo Citro

# For this project, this is the second part where I am comparing the Runge-Kutta method against
# ODEint and comparing the performance

# Implementation approach:
# - developed an ODE function using scypi.odeint (outlined in documentation)
# - developed a Runge-Kutta function based on the formulas used in class (outlined in documentation)
# - Solved both functions using a differential equation and graphed the results against each other
# - Recorded the time spent solving each function and compared them
# - Found the error percentage at each point of the functions
# - Found the total error percentage between the functions

# Packages used: time, numpy, matplotlib, scipy
import time                         # import time to use for performance analysis
import numpy as np                  # import numpy for array space
import matplotlib.pyplot as plt     # import matplotlib for graphing functions
from scipy.integrate import odeint  # import scipy to use the ordinary differencial equation integral function

# Ordinary Differential Equation
def dy_dx(y, x):                    # takes inputs y and x
    return y / (np.exp(x) - 1)      # returns the function y/(e^x - 1)

# Runge-Kutta Forumla based off the formula used in class
def rungeKutta(y, x, h):                                # takes y, x, and h inputs
    k1 = dy_dx(y, x)                                    # solves k1 using the differential equation function
    k2 = dy_dx(y + ((0.5 * h) * k1), x + 0.5 * h)       # solves k2 based on the answer from k1 using the differential equation function
    k3 = dy_dx(y + ((0.5 * h) * k2), x + 0.5 * h)       # solves k3 based on the answer from k2 using the differential equation function
    k4 = dy_dx(y + (h * k3), x + h)                     # solves k4 based on the answer from k3 using the differential equation function

    t4 = (1.0 / 6.0) * (k1 + (2 * k2) + (2 * k3) + k4)  # solves for t4 by taking a 6th of k1 + 2*k2 + 2*k3 + k4

    y = y + (t4 * h)                                    # solves for y by taking the initial y value and adding it to t4*h
    return y                                            # returns y

# initial variable values
y = 5           # initial y value of 5 as outlined in the assignment
x = 1           # initial x value of 1 as outlined in the assignment
h = 0.02        # initial h value of 0.02 as outlined in the assignment
n = 2000        # initial n value of 2000 chosen between 1000 or 2000 as outlined in the assignment

# Runge-Kutta x and y arrays 
xsr = []        # x-space runge-kutta array to store the values of x for the runge-kutta function
ysr = []        # y-space runge-kutta array to store the values of y for the runge-kutta function

# ODEint x and y arrays, solution, and time analysis
tso = time.time()                               # time start for ODEint function solution

xso = np.linspace(1, (int)(n * h) + 1, n)       # x-space ODEint from 1 to n*h (40) plus 1 with a step size of n
yso = odeint(dy_dx, y, xso)                     # y-space ODEint useing the odeint function from scicpy to find the y-space

teo = time.time()                               # time end fore ODEint function solution
tto = teo - tso                                 # total time the ODEint function to solve

# graphing ODEint
plt.title("ODE Function Analysis")          # set the title of the graph
plt.xlabel("x")                                             # set the x label on the graph
plt.ylabel("y")                                             # set the y label on the graph
plt.plot(xso, yso, 'r-', label = "ODEint", linewidth = 2)   # set the ODE line to be red and label it
plt.legend()                                                # shows the legend on the graph
plt.show()                                                  # displays the graph

# Runge-Kutta solution and time analysis
tsr = time.time()                               # time start for runge-kutta function solution

for i in range(0, n):                           # for loop to run the runge-kutta function n number of times
    xsr.append(x)                               # append the x value to the x-space runge-kutta array
    ysr.append(y)                               # append the y value to the y-space runge-kutta array
    y = rungeKutta(y, x, h)                     # update the y value using the rungeKutta function
    x += h                                      # update the x value by moving one step forward (0.02)

ter = time.time()                               # time end for runge-kutta function solution
ttr = ter - tsr                                 # total time the runge-kutta function to solve

td = ttr - tto                                  # time difference between ODEint and runge-kutta function

# graphing runge-kutta
plt.title("Runge-Kutta Function Analysis")          # set the title of the graph
plt.xlabel("x")                                             # set the x label on the graph
plt.ylabel("y")                                             # set the y label on the graph
plt.plot(xsr, ysr, 'b-', label = "Runge Kutta")             # set the runge-kutta to be blue and label it
plt.legend()                                                # shows the legend on the graph
plt.show()                                                  # displays the graph

# solutions
print("\nODEint Solution:            ", yso[-1])    # ODEint function solution
print("Runge-Kutta Solution:        ", ysr[-1])     # Runge-Kutta function solution

# Print statement for time difference
print("\nODEint Time:         ", tto)               # print the ODEint time
print("Runge Kutta Time:    ", ttr)                 # print the runge-kutta time
print("ODEint is ", td, " seconds faster\n\n")      # print the difference between ODEint and runge-kutta

# error calculation
error = 0                                                           # initial error value of 0
errorRange = []                                                     # array to store error over xn
errorSpace = np.linspace(1, (int)(n * h) + 1, 41)                   # error space for error analysis
for i in range((int)(n * h) + 1):                                   # for loop to run through every x values
    error += (np.abs(ysr[i] - yso[i])/yso[i]) * 100                 # sum all the error values using the percentage error formula
    errorRange.append((np.abs(ysr[i] - yso[i])/yso[i]) * 100)
    print("Percent Error at x =", i, ":", (np.abs(ysr[i] - yso[i])/yso[i]) * 100)   # print error at each x value

print("\nAverage Error Percent:", error/((int)(n * h) + 1), "\n")     # print the total error divided by the total number of x values

# graphing error
plt.title("Error Analysis")
plt.xlabel("xn")
plt.ylabel("error")
plt.plot(errorSpace, errorRange, label = "Error over Xn")
plt.legend()
plt.show()

# graphing both functions
plt.title("Runge-Kutta and ODE Function Analysis")          # set the title of the graph
plt.xlabel("x")                                             # set the x label on the graph
plt.ylabel("y")                                             # set the y label on the graph
plt.plot(xso, yso, 'r-', label = "ODEint", linewidth = 2)   # set the ODE line to be red and label it
plt.plot(xsr, ysr, 'b-', label = "Runge Kutta")             # set the runge-kutta to be blue and label it
plt.legend()                                                # shows the legend on the graph
plt.show()                                                  # displays the graph