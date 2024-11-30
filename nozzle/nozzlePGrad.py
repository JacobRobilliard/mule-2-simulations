import math

import matplotlib
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def nozzPGrad(r, x, k, R_T, Po):
    # Throat Area
    Astar = np.pi * (R_T ** 2)

    # Area Array
    A = np.zeros((len(r), 1))

    # Populating it
    for i in range(len(r)):
        A[i] = np.pi * r[i] ** 2

    # Mach Number Array
    Ma = np.zeros((len(r), 1))
    MaLast = 0.1

    MaShock = np.zeros((len(r), 1))

    # Pressure gradient array
    P = np.zeros((len(r), 1))

    PShock = np.zeros((len(r), 1))

    # Flag to switch assumption for solving once past throat
    flag = True

    # Now we iterate over the nozzle and calculate the Mach Number and pressure at each discrete point
    # This is based on the contour generated earlier, so it will have more "resolution" in the contour area
    for i in range(0, len(r)):

        # Calculating Mach number at each point in nozzle, has two solutions, one for subsonic and one for super
        def AA(x):
            return (1 / x) * ((1 + ((k - 1) * (x ** 2) / 2)) / ((k + 1) / 2)) ** ((k + 1) / 2 / (k - 1)) - (
                        A[i] / Astar)

        # Once it passes the throat we assume it's greater than one and adjust the initial guess
        if x[i] > 0 and flag:
            MaLast = 1.1
            flag = False

        Ma[i] = fsolve(AA, MaLast)

        # Change estimate to be last value as it is continuous
        MaLast = Ma[i]

        # We then calculate the pressure at that point based on the Mach Number and the stagnation pressure of the
        # combustion chamber, will be in whatever units that is given in
        P[i] = Po * (1 + (k - 1) * Ma[i] ** 2 / 2) ** (-k / (k - 1))

        # Estimating shock Mach Number at each point
        def shockMa(x):
            return ((k - 1) * Ma[i] ** 2 + 2) / (2 * k * Ma[i] ** 2 - (k - 1)) - x ** 2

        # Calculating pressure after shock at each point
        if x[i] < 0:
            MaShock[i] = np.nan
            PShock[i] = np.nan
        else:
            MaShock[i] = fsolve(shockMa, 0.5)
            PShock[i] = P[i] * (1 + (k * Ma[i] ** 2)) / (1 + (k * MaShock[i] ** 2))
            # Removing weak shocks from consideration
            if PShock[i] / P[i] <= 2:
                PShock[i] = np.nan

    fitP = np.poly1d(np.ravel(np.polyfit(np.ravel(x), P, 32)))
    #CS = CubicSpline(x.flatten(), P.flatten())
    #print(CS)
    xp = np.linspace(x[0], x[-1], num=10001)

    # Plotting stuff, change as needed
    Mach, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(x, r, 'k')
    ax1.label_outer()
    ax1.set_title("Nozzle Radius [m] vs Length [m]")
    ax2.plot(x, Ma, 'r', label="Ma")
    ax2.plot(x, MaShock, '--r', label="Shock Ma")
    ax2.legend()
    ax2.set_title("Mach Number vs Length [m]")
    ax3.plot(x, P, 'b', label="Flow Pressure")
    ax3.plot(x, PShock, '--b', label="Shock Pressure")
    ax3.legend()
    ax3.set_title("Pressure [psi] vs Length [m]")
    ax4.plot(xp, fitP(xp), label="Poly")
    # ax4.plot(x, P, '--', label="P")
    # ax4.legend()
    ax4.set_title("Pressure Spline Fit [psi] vs Length [m]")
    plt.show()


def Contour_Equation(R_throat, Expand, k, c, T, Rc):
    # Define variable x
    x = 0

    # Define Constants
    theta_wall = 23 * 3.14159265 / 180  # Degrees
    R_D = 0.2 * R_throat
    R_U = 0.5 * R_throat  # can be from 0.5-1.5*R_throat
    R_E = math.sqrt(Expand * R_throat ** 2)
    theta_e = 10.5 * 3.14159265 / 180  # angle of the trailing edge of the cone

    # Calculate Nozzle Equation Coefficients
    Ex = 0.8 * ((math.sqrt(Expand) - 1) * R_throat) / math.tan(15 / 180 * 3.14159265)
    print("Ex: ", Ex)
    Ey = R_E
    print("Ey: ", Ey)
    Nx = R_D * math.cos(theta_wall - 3.14159265 / 2)
    print("Nx: ", Nx)
    Ny = R_D * math.sin(theta_wall - 3.14159265 / 2) + R_D + R_throat
    print("Ny: ", Ny)
    Qx = (Ey - math.tan(theta_e) * Ex - Ny + math.tan(theta_wall) * Nx) / (math.tan(theta_wall) - math.tan(theta_e))
    print("Qx: ", Qx)
    Qy = (math.tan(theta_wall) * (R_E - math.tan(theta_e) * Ex) - math.tan(theta_e) * (
            Ny - math.tan(theta_wall) * Nx)) / (math.tan(theta_wall) - math.tan(theta_e))
    print("Qy: ", Qy)

    # Calculate the Contour Equation
    # Create x array for plotting
    t = (np.linspace(0, 1, num=1001))

    # create empty arrays to use later
    x = np.zeros((len(t), 1))
    r = np.zeros((len(t), 1))

    # Loop to copy fill the t and r arrays
    i = 0
    while i < len(t):
        # Arc length as a function of x
        x[i] = Nx * (1 - t[i]) ** 2 + 2 * Qx * (t[i] - t[i] ** 2) + Ex * t[i] ** 2
        # Parametric Equation for Arc Length
        r[i] = Ny * (1 - t[i]) ** 2 + 2 * Qy * (t[i] - t[i] ** 2) + Ey * t[i] ** 2
        i += 1

    # Radiused Diverging Section
    x1 = np.linspace(0, x[0], num=200, endpoint=False)
    r1 = np.zeros((len(x1), 1))

    for i in range(len(x1)):
        def circDiv(y):
            return (x1[i] ** 2) + (y - R_throat * 1.45) ** 2 - (R_throat * 0.45) ** 2

        r1[i] = fsolve(circDiv, R_throat)

    # Shifting Contour down to match radiused diverging section
    # A bit jank, but I think it's fine
    def shift(y):
        return (x[0] ** 2) + (y - R_throat * 1.45) ** 2 - (R_throat * 0.45) ** 2

    dif = r[0] - fsolve(shift, r[0])

    #r = r - dif

    # Radiused Converging Section
    x2 = np.linspace(-1.5 * R_throat * math.cos(math.radians(45)), x1[0], endpoint=False)
    r2 = np.zeros((len(x2), 1))

    for i in range(len(x2)):
        def circConv(y):
            return (x2[i] ** 2) + (y - R_throat * 2.5) ** 2 - (R_throat * 1.5) ** 2

        r2[i] = fsolve(circConv, R_throat)

    # Straight Convering Section
    x3 = np.linspace(-(Rc - r2[0]) + x2[0], x2[0], endpoint=False)
    r3 = np.linspace(Rc, r2[0], endpoint=False)

    # Adding all sections together
    x = np.concatenate((x3, x2, x1, x))
    r = np.concatenate((r3, r2, r1, r))

    return x, r


# This function is called with the values: Contour_Equation(R_throat, Expand, k, c, T )
# When porting over to the full python code, just modify the function inputs to match the variables used in the code
throat = 0.0154622946393
Ex = 4.8688
k = 1.242
c = 2.799
T = 2842.0
Pc = 400
Rc = 45.75 / 1000

x, r = Contour_Equation(throat, Ex, k, c, T, Rc)
nozzPGrad(r, x, k, throat, Pc)
