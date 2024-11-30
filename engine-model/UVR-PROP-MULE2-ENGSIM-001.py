#   Mule-2 Engine Simulation | UVR-PROP-MULE2-ENGSIM-001.py
#   Created for: UVR Propulsion
#
#   Created by: Jacob Robilliard, Lead Engineer
#   Date created: 25/11//2024
#
#   *NOT FOR PUBLIC RELEASE*
#
#       ##READ ME##
#
#   This software uses NASA's Chemical Equilibrium with Applications code
#   Refer to relevant documentation to setup and install on your own device
#
#   An example command to run CEA is as follows:
#   
#   s = CEARun.get_full_cea_output( Pc=1000.0, MR=6.0, eps=40.0, short_output=1)
#   
#   Where Pc = combustion chamber pressure in [PSI], MR = O/F ratio (unitless), eps = expansion ratio (unitless)

############################################################################
# Import libraries
import sys
import numpy as np
import scipy
import math

import scipy.constants
import scipy.optimize
from scipy.optimize import fsolve 
from scipy.optimize import shgo

import matplotlib.pyplot as plt

sys.path.insert(0,'/Users/jacobr/Library/Python/3.9/bin') # This path can be changed if necessary
from rocketcea.cea_obj_w_units import CEA_Obj # type: ignore
from rocketcea.units import add_user_units  # type: ignore # Allows for the addition of custom units
from rocketcea.cea_obj import add_new_fuel # type: ignore # Allows for the addition of Paraffin as a fuel

# Convert CEA Units
add_user_units( 'BTU/lbm', 'J/Kg', 2326 ) # multiplier = user units / default units

card_str = """
fuel Paraffin  C 31 H 64  wt%=100
h,cal=-444695     t(k)=298.15
""" # Defines the properties of Paraffin as a fuel
add_new_fuel( 'Paraffin', card_str) # Adds Paraffin to the fuel database

# Initialize CEA Object
CEARun = CEA_Obj( oxName='N2O', fuelName='Paraffin',
                 
                            isp_units            = 'sec',         # N-s/kg, m/s, km/s
                            cstar_units          = 'm/sec',       # ft/s
                            pressure_units       = 'Pa',          # MPa, KPa, Psia, Bar, Atm, Torr
                            temperature_units    = 'K',           # degR, C, F
                            sonic_velocity_units = 'm/sec',       # ft/s
                            enthalpy_units       = 'J/Kg',        # BTU/lbm, J/Kg
                            density_units        = 'kg/m^3',      # g/cc, sg, lbm/cuft
                            specific_heat_units  = 'J/kg-K',      # kJ/kg-K, cal/g-C, BTU/lbm degR
                            viscosity_units      = 'millipoise',  # lbf-sec/sqin, lbf-sec/sqft, lbm/ft-sec, poise, centipoise
                            thermal_cond_units   = 'mcal/cm-K-s'  # millical/cm-degK-sec, BTU/hr-ft-degF, BTU/s-in-degF, cal/s-cm-degC, W/cm-degC) 
                    )

# Define ambient/universal constants
g = scipy.constants.g
pi = scipy.constants.pi

# End of Initialization commands
############################################################################ 
# Function Definitions
def BurnSimulation(Optimized_input, Constant_values):
    Engine_config = Constant_values[0]
    Optimal_expansion = Constant_values[1]
    # Define system conditions, Do Not Edit
    rho_f = Engine_config[2] # density of paraffin, [kg/m^3]
    n = Engine_config[3][1] # Regression Rate Flux Exponent, [-]
    m = Engine_config[3][2] # Regression Rate Length Exponent, [-]
    a_0 = Engine_config[3][0] # Regression Rate Coefficient, [m/s]
    #G_ox_i = Engine_config[2][1] # maximum recomended oxidizer flux for liquid oxidizers, [kg/m^2*s] (Humble)
    E_n = Optimal_expansion[1] # Nozzle Isentropic Efficency, [-] 
    epsilon = Optimal_expansion[0]
    P_c = Engine_config[0] # Engine chamber pressure, [Pa]
    P_c_i = P_c
    Ap_f = Engine_config[1] # Final port area - constrained by Pro98 Combustion chamber, [m^2]

    m_dot_ox = Optimized_input[0]
    OF_i = Optimized_input[1]
    G_ox_i = Optimized_input[2]
    ############################################################################ 
    # Simulation Algorithm (Table 7.9) 
    ############################################################################ 
    t = 0 #simulation start, [s]

    # Initial Thermochemistry
    c_star_i = CEARun.get_Cstar(Pc=P_c, MR=OF_i) #[m/s]

    # Inital mass and flow properties
    m_dot_f = m_dot_ox/OF_i #[kg/s]
    m_dot_p = m_dot_ox+m_dot_f #[kg/s]
    A_t = c_star_i*m_dot_p/P_c #[m^2]
    A_e = epsilon*A_t

    # Fuel grain, pre and post combustor dimensions, and diameters
    Ap = m_dot_ox/G_ox_i # Initial port area - gets updated by burn simulation below, [m^2]
    Dp = 2*math.sqrt(Ap/pi) #Initial port diameter - gets updated by nurn simualtion below, [m]
    Dp_i = Dp # Create variable Dp_i, [m]
    L_g = (m_dot_f/(rho_f*a_0*G_ox_i**n*pi*Dp_i))**(1/(1+m)) # Length of the fuel grain, [m]
    L_pre = 0.5*Dp_f # Length of the pre combustion spacer, [m]
    L_post = 0.7*Dp_f # Length of the post combustion spacer, [m]
    L_total = L_g + L_pre + L_post # Total length required for combustion chamber, [m]

    # Beginning of performance simulation
    Sp = 2*math.sqrt(Ap*pi)*L_g #[m^2]

    y_OF = []
    y_F = []
    y_Isp = []
    x_t = []

    t = 0 #[s]
    while Ap <= Ap_f:   
        x_t.append(t)

        # Action 1: Determine fuel regression rate
        G_ox = m_dot_ox/Ap
        r_dot = a_0*G_ox**n #[m/s]
    
        # Action 2: Determine total mass flow rate
        m_dot_f = r_dot*rho_f*Sp #[kg/s]    
        m_dot_p = m_dot_f + m_dot_ox #[kg/s]
        
        #Action 3: Determine Thermochemistry
        OF = m_dot_ox/m_dot_f
        y_OF.append(OF)
        CstrTc_exitMWGam = CEARun.get_IvacCstrTc_exitMwGam(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)
        gamma = CstrTc_exitMWGam[4]
        cstar = CstrTc_exitMWGam[1]
        M = CstrTc_exitMWGam[3]

        # Action 4: Determine chamber, nozzle, and engine performance
        P_c = m_dot_p*cstar/A_t #[Pa]
        Me = CEARun.get_MachNumber(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)
        T_e = CEARun.get_Temperatures(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)[2]
        R = 8314.41/M
        v_e = Me*math.sqrt(gamma*R*T_e)
        F = E_n*(m_dot_p*v_e)
        y_F.append(F)
        Isp = F/(m_dot_p*g)
        y_Isp.append(Isp)
        
        # Action 5: Iterate
        t = t + 0.05 #[s]

        # Action 6: Update grain geometry
        Dp = ((a_0)*(4*n+2)*(4*m_dot_ox/pi)**n*t + Dp_i**(2*n+1))**(1/(2*n+1)) #[m]
        Ap = pi*(Dp/2)**2 #[m^2]
        Sp = 2*math.sqrt(Ap*pi)*L_g #[m^2]

    # Engine performance output
    OF_avg = sum(y_OF)/len(y_OF)
    F_avg = sum(y_F)/len(y_F)
    Max_Thrust = max(y_F)
    Total_impulse = F_avg*t
    Isp_avg = sum(y_Isp)/len(y_Isp)
    m_ox = m_dot_ox*t
    m_f = m_ox/OF_avg
    
    print('Average Thrust: ',"{:.1f}".format(F_avg),'[N]\n'
        'Peak Thrust: ',"{:.1f}".format(Max_Thrust),'[N]\n'
        'Total Impulse ',"{:.1f}".format(Total_impulse),'[Ns]\n'
        'Average Isp: ',"{:.1f}".format(Isp_avg),'[s]\n'
        'Average OF: ',"{:.2f}".format(OF_avg),'[-]\n'
        'Burn time: ',"{:.2f}".format(t),'[s]\n'
        '\n''Fuel grain length: ',"{:.3f}".format(L_g),'[m]\n'
        'Pre Combustor Length: ',"{:.3f}".format(L_pre),'[m]\n'
        'Post Combustor Length: ',"{:.3f}".format(L_post),'[m]\n'
        'Total Combustion Length',"{:.3f}".format(L_total),'[m]\n'
        'Oxidizer Flow Rate: ',"{:.2f}".format(m_dot_ox),'[kg/s]\n'
        'N2O Mass: ',"{:.3f}".format(m_ox),'[kg]\n'
        'Paraffin Mass: ',"{:.3f}".format(m_f),'[kg]\n'
        'Other Engine Parameters: \n'
        '   Nozzle Expansion Ratio: ',"{:.4f}".format(epsilon),'[-]\n'
        '   Nozzle Throat Area: ', "{:.8f}".format(A_t),'[m^2]\n'
        '   Nozzle Exit Area: ', "{:.8f}".format(A_e),'[m^2]\n'
        '   Chamber Pressure: ',"{:.1f}".format(P_c_i/6894.76),'[Psia]\n'
        '   Port Initial Diameter: ',"{:.4f}".format(Dp_i),'[m]\n')
    
    # Output Plots
    fig = plt.figure()
    gs = fig.add_gridspec(1,3, wspace=1)
    axs = gs.subplots(sharex=True)
    fig.suptitle('Mule-2 Performance Simulation')
    axs[0].plot(x_t, y_F)
    axs[0].set_title("Thrust vs Time")
    axs[0].set(xlabel='Time [s]', ylabel='Thrust [N]')
    axs[1].plot(x_t, y_Isp)
    axs[1].set_title("Specific Impulse vs Time")
    axs[1].set(xlabel='Time [s]', ylabel='Specific Impulse [s]')
    axs[2].plot(x_t, y_OF)
    axs[2].set_title("O/F Ratio vs Time")
    axs[2].set(xlabel='Time [s]', ylabel='O/F Ratio')

    plt.show()
    
    return 0

def Optimize_Param(params, Engine_config, Optimal_expansion, Design_Requirements):
    # Define system conditions, Do Not Edit
    rho_f = Engine_config[2]
    n = Engine_config[3][1] # Regression Rate Flux Exponent, [-]
    m = Engine_config[3][2] # Regression Rate Length Exponent, [-]
    a_0 = Engine_config[3][0] # Regression Rate Coefficient, [m/s]
    E_n = Optimal_expansion[1] # Nozzle Isentropic Efficency, [-] 
    epsilon = Optimal_expansion[0]
    P_c = Engine_config[0] # Engine chamber pressure, [Pa]
    Ap_f = Engine_config[1] # Final port area - constrained by Pro98 Combustion chamber, [m^2]

    # Vaiables editable by Opyimize_Param()
    m_dot_ox = params[0] # oxidizer flow rate (constant), [kg/s]
    OF = params[1] # OF Ratio, [-]
    G_ox_i = params[2]

    # Lists to store thrust, and specific impulse
    y_F = []
    y_Isp = []

    # Requirements
    F_desired = Design_Requirements[0]
    Isp_Desired = Design_Requirements[1]
    Desired_Burn_Time = Design_Requirements[2]

    # Begin Burn simulation
    Ap = m_dot_ox/G_ox_i # Initial port area - gets updated by burn simulation below, [m^2]
    Dp = 2*math.sqrt(Ap/pi) #Initial port diameter - gets updated by nurn simualtion below, [m]
    m_dot_f = m_dot_ox/OF #Initial port diameter - gets updated by nurn simualtion below, [kg/s]
    m_dot_p = m_dot_ox + m_dot_f #Initial port diameter - gets updated by nurn simualtion below, [kg/s]
    L_g = (m_dot_f/(rho_f*a_0*G_ox_i**n*pi*Dp))**(1/(1+m)) # Length of the fuel grain, [m]
    Sp = 2*math.sqrt(Ap*pi)*L_g #[m^2]

    # Initial Thermochemistry - sizes throat
    c_star_i = CEARun.get_Cstar(Pc=P_c, MR=OF) # Charictaristic velocity, [m/s]
    A_t = c_star_i*m_dot_p/P_c # Throat area, [m^2]

    t = 0
    Dp_i = Dp # Create a variable to store the initial port diameter
    while Ap <= Ap_f:   
        # Action 1: Determine fuel regression rate
        G_ox = m_dot_ox/Ap
        r_dot = a_0*G_ox**n #[m/s]

        # Action 2: Determine total mass flow rate
        m_dot_f = r_dot*rho_f*Sp #[kg/s]    
        m_dot_p = m_dot_f + m_dot_ox #[kg/s]
        
        #Action 3: Determine Thermochemistry
        OF = m_dot_ox/m_dot_f
        CstrTc_exitMWGam = CEARun.get_IvacCstrTc_exitMwGam(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)
        gamma = CstrTc_exitMWGam[4]
        cstar = CstrTc_exitMWGam[1]
        M = CstrTc_exitMWGam[3]

        # Action 4: Determine chamber, nozzle, and engine performance
        P_c = m_dot_p*cstar/A_t #[Pa]
        Me = CEARun.get_MachNumber(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)
        T_e = CEARun.get_Temperatures(Pc=P_c, MR=OF, eps=epsilon, frozen=0, frozenAtThroat=0)[2]
        R = 8314.41/M
        v_e = Me*math.sqrt(gamma*R*T_e)
        F = E_n*(m_dot_p*v_e)
        y_F.append(F)
        Isp = F/(m_dot_p*g)
        y_Isp.append(Isp)

        # Action 5: Iterate
        t = t + .1 #[s]

        # Action 6: Update grain geometry
        Dp = ((a_0)*(4*n+2)*(4*m_dot_ox/pi)**n*t + Dp_i**(2*n+1))**(1/(2*n+1)) #[m]
        Ap = pi*(Dp/2)**2 #[m^2]
        Sp = 2*math.sqrt(Ap*pi)*L_g #[m^2]

    # Engine performance output
    if( len(y_F) != 0 and len(y_Isp) != 0):
        F_avg = sum(y_F)/len(y_F)
        Isp_avg = sum(y_Isp)/len(y_Isp)
    else:
        F_avg = 0
        Isp_avg = 0

    return  math.sqrt( (F_avg-F_desired)**2 + (Isp_avg-Isp_Desired)**2 + 1000*(t-Desired_Burn_Time)**2 )
# End of Fucntion Definitions
############################################################################
# Define system conditions, These inputs should not be changed
rho_f = 930 # density of paraffin, [kg/m^3]
n = 0.555 # Regression Rate Flux Exponent, [-]
m = 0 # Regression Rate Length Exponent, [-]
a_0 = 0.132 # Regression Rate Coefficient, [mm/s]
P_a = 97120.77 # Ambient pressure at test site, [Pa]
E_n = 0.98 # Nozzle Isentropic Efficency, [-]

# Define Design Decisions 
P_c = 400 # Combustion Chamber Pressure, [Psia] 
Dp_f = 3.375  # constraint imposed by the inner diameter of a Pro98 phenolic liner, [in]
Design_Requirements = (3000, 225, 5) #Requirements for a sucessful engine: 3000 [N], 225 [s], 5 [s]

# Input value modification
a_0 = a_0/1000 #[m/s]
fuel_data = (rho_f) # Tuple of fuel data (rho_f, G_ox_i)
regression_data = (a_0, n, m) #Tuple of resgression coefficients (a_0, n, m)

P_c = P_c*6894.76 #[Pa]
Dp_f = Dp_f/39.37 #[m]
Ap_f = pi*(Dp_f/2)**2 # [m^2]

Engine_config = (P_c, Ap_f, fuel_data, regression_data)
# End of Inputs
############################################################################ 
# Calculation of optimal OF Ratio based on fuel inputs
OF_test = np.arange(1.0, 12.0, 0.1)
Isp_test = []
for OF in OF_test:
    Isp = CEARun.get_Isp(Pc=P_c, MR=OF, eps=5.3)
    Isp_test.append(Isp)
OF_index = Isp_test.index(max(Isp_test))
OF_optimal = OF_test[OF_index]

# ISP vs expansion ratio (valid for atmospheric operation, Pe = Pa) (Optimal/frozen flow conditions)
epsilon = CEARun.get_eps_at_PcOvPe(Pc=P_c, MR=OF, PcOvPe=(P_c/P_a), frozen=0, frozenAtThroat=0)

Optimal_expansion = (epsilon,E_n) # Tuple for optimal expansion data

# Engine Simulation V 1.0
bnds = ((0.01, 3),(0.01,8), (0.01, 750))
Constant_values = (Engine_config, Optimal_expansion, Design_Requirements) # Tuple of constant values passed to functions Optimize_Param() and BurnSimulation()
result = shgo(Optimize_Param, args=Constant_values, bounds=bnds, n=64, iters=3, sampling_method='sobol')
m_dot_ox = result.x[0]
OF_i = result.x[1]
G_ox_i = result.x[2]
Optimized_input = (m_dot_ox, OF_i, G_ox_i)

print('Optimized Inout: \n'
        'Initial G_ox: ',"{:.1f}".format(G_ox_i),' [kg/m^2*s]\n'
        'Initial OF ratio: ',"{:.1f}".format(OF_i),' [-]\n'
        'Oxidizer Mass Flow: ',"{:.1f}".format(m_dot_ox),' [kg/s]\n'
        'See below for complete simulation:\n')
BurnSimulation(Optimized_input, Constant_values)