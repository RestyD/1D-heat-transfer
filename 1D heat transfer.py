'''
This program corresponds to the problem 3.1, one dimentional heat transfer, from the book "Computational Fluid Dynamics for Engineers" written for J. Xamán
It aims to determinate the variation in temperature in one direction T(x), in a lead bar with homogeneous thermal conductivity and specific heat 
(λ = 35 W / m °C. Cp = 130 J/Kg°C). Consider that only heat transfer occurs and there is no generation of the variable. The bar lenght is 1 m, and
it is constraint to 1st class boundary conditions. The analytic solution to this problem is: T(x) = [(T_B - T_A) / H_x ] * x + T_A.

                |---------------------- Hx = 1 m -----------------------|
                 _______________________________________________________
    T_A = 0 °C  |                                                       |  T_B = 100 °C
                |_______________________________________________________|
                
                y
                |__ x
                
Resty Durán February 15th 2024
'''
# import libraries
import numpy as np
import matplotlib.pyplot as plt

# Domain dimensions and mesh nodes 
Hx = 1.0                    # Domain size x direction
Hy = 1.0                    # Domain size y direction
nx = 5                      # Number of cells in x-direction
ny = 5                      # Number of cells in y-direction
Nx = nx + 2                 # Number of nodes in x-direction
Ny = ny + 2                 # Number of nodes in y-direction

# Material propierties
lamda = 35                  # Thermal conductivity (W / m °C)
cp = 130                    # Specific heat (J/ kg C)
gamma = lamda / cp          # diffusive transport
sp = 0                      # Source term

# Solution process
NI= 140                     # Number of iterations
convergence = 1E-6          # Convergence 

def meshing():
    ''' Calculate the nodes coordinates x, y '''
    dx = Hx / (Nx - 2)
    dy = Hy / (Ny - 2)
    x = np.hstack(([0], [dx/2 + i*dx for i in range(nx)], [Hx]))
    y = np.hstack(([0], [dy/2 + i*dy for i in range(ny)], [Hy]))
    X, Y = np.meshgrid(x, y)
    return X, Y, dx, dy

def show_mesh():
    ''' Plot results and draw mesh and nodes'''
    X, Y, dx, dy = meshing()
    plt.xticks(np.linspace(0, Hx, nx + 1))
    plt.yticks(np.linspace(0, Hy, ny + 1))
    plt.xlim(0, Hx)
    plt.ylim(0, Hy)
    plt.title(' Computational mesh')
    plt.xlabel('x direction (m)')
    plt.ylabel('y direction (m)')
    # This is for avoiding ticks overlaping
    #plt.gca().set_xticklabels([])
    #plt.gca().set_yticklabels([])
    
    plt.scatter(X, Y, color = 'black')
    plt.grid()
    plt.show()
    
def solution(NI):
    ''' Numerical solution of the problem (Jabobi's method)'''
    X, Y, dx, dy = meshing()
    x= X[0]
    residuals = []

    # Step 1: Suppose a distribution of T_P
    T = np.array([0.0 if (ti > 0 and ti < len(x) - 1) else 50.0 for ti in range(len(x))])
    Tn = np.copy(T)
    
    # Step 2: Calculate the coefficients of all the nodes a_W, a_E, a_P and b
    aW = np.array([gamma / abs(x[i] - x[i - 1]) if (i > 0 and i < len(x) - 1) else 0 for i in range(len(x))])
    aE = np.array([gamma / abs(x[i] - x[i + 1]) if (i > 0 and i < len(x) - 1) else 0 for i in range(len(x))])
    aP = aW + aE - sp * dx
    b  = np.zeros(len(x))
    aP[ 0] = 1
    aP[-1] = 1
    b[  0] = 0
    b[ -1] = 100

    # Step 3 Calculate T_p in all nodes using the generative equation T_P = (a_W * T_W + a_E * T_E) / a_P
    while(NI > 0):
        for i in range(len(Tn)):
            # Boundary node (aW = aE = 0, aP = 1, b = T = 0)
            if i == 0:
                Tn[i] = b[i] / aP[i]

            # Boundary node (aW = aE = 0, aP = 1, b = T = 100)
            elif i >= len(Tn) - 1:
                Tn[i] = b[i] / aP[i]

            # Internal nodes  
            elif i > 0 and i < len(Tn) - 1:
                Tn[i] = (aW[i] * T[i - 1] + aE[i] * T[i + 1] + b[i]) / aP[i]
        
        # Calculation of residuals
        rT = 0
        for i in np.arange(1, len(Tn) - 1):
            rT =rT + abs(aP[i] * Tn[i] - (aW[i] * Tn[i - 1] + aE[i] * Tn[i + 1] + b[i]))
        residuals.append(rT)
        print(f'RESIDUALS Iteration {len(residuals)}, T = {residuals[-1]}')

        # Update the variable T
        T = np.copy(Tn)

        # Convergence criteria
        if rT <= convergence:
            break

        # Update iteration
        NI -= 1
    return T, residuals

def results(T, residuals):
    ''' Plot residuals '''
    plt.plot(np.arange(0, len(residuals)), residuals, color = 'black')
    plt.xlabel('iterations')
    plt.ylabel('residuals')
    plt.show()
    
    ''' Plot results and draw mesh and nodes'''
    X, Y, dx, dy = meshing()
    plt.contourf(X, Y, np.tile(T, (len(Y), 1)), cmap = 'jet', levels = 50)
    plt.colorbar()
    plt.xticks(np.linspace(0, Hx, nx + 1))
    plt.yticks(np.linspace(0, Hy, ny + 1))
    plt.xlim(0, Hx)
    plt.ylim(0, Hy)
    plt.title(' Temperature field (°C)')
    plt.xlabel('x direction (m)')
    plt.ylabel('y direction (m)')
    # This is for avoiding ticks overlaping
    #plt.gca().set_xticklabels([])
    #plt.gca().set_yticklabels([])
    
    plt.scatter(X, Y, color = 'black')
    plt.grid()
    plt.show()

show_mesh()
T, residuals = solution(NI)
print(T)
results(T, residuals)
