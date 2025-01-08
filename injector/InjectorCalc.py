#   Mule-2 Injector Calculator | InjectorCalc.py
#   Created for UVR Propulsion
#
#   Created by Alex Badzio-George
#   October 2024
#   
#   This program requires the CoolProp library
#
#   Theory based on "Mass Flow Rate and Isolation Characteristics of Injectors
#                   for Use with Self-Pressurizing Oxidizers in Hybrid Rockets" 
#          by B. Waxman, J. E. Zimmerman, B. J. Cantwell, & G. G. Zilliac
#          found at https://ntrs.nasa.gov/api/citations/20190001326/downloads/20190001326.pdf               
#   
##################################################################
import math
import CoolProp
from CoolProp.CoolProp import PropsSI
import os
import sys


class InjectorCalc:

    #Default Units:
    #Pressure: Pa
    #Enthalpy: J/kg
    #Entropy: J/kgK

    fluid:str = 'NitrousOxide'

    #Primary Functions
    def getExitState(T1: float, p1: float, p2: float, X = -1) -> list:
        
        isentropicEff: float = 0.7
        if isentropicEff > 1:
            print("Isentropic Efficiency cannot be greater than 1")

        try:
            if X == -1:
                h1 = PropsSI('H', 'T', T1, 'P', p1, InjectorCalc.fluid)
                s1 = PropsSI('S', 'T', T1, 'P', p1, InjectorCalc.fluid)
            else:
                h1 = PropsSI('H', 'T', T1, 'Q', X, InjectorCalc.fluid)
                s1 = PropsSI('S', 'T', T1, 'Q', X, InjectorCalc.fluid)
        except ValueError as e:
            h1 = PropsSI('H', 'T', T1, 'Q', 0, InjectorCalc.fluid)
            s1 = PropsSI('S', 'T', T1, 'Q', 0, InjectorCalc.fluid)

        s2s = s1; #Isentropic process to point 2s

        h2s = PropsSI('H', 'S', s2s, 'P', p2, InjectorCalc.fluid)

        h2 = h1 + isentropicEff*(h2s - h1)

        T2 = PropsSI('T', 'H', h2, 'P', p2, InjectorCalc.fluid)

        try:
            psat2 = PropsSI('P', 'T', T2, 'Q', 1, InjectorCalc.fluid)
        except ValueError as e:
            print("Supercritical Vapour - Models not Valid")
            sys.exit()
  
        hf = PropsSI('H', 'T', T2, 'Q', 0, InjectorCalc.fluid)
        hg = PropsSI('H', 'T', T2, 'Q', 1, InjectorCalc.fluid)
        X = (h2 - hf)/(hg-hf)

        v2 = math.sqrt(2*(h1-h2))

        return [T1, p1, T2, p2, X, v2]
    
    def printState(T, p, X = -1):
        
        if X == -1:
            try:
                h = PropsSI('H', 'T', T, 'P', p, InjectorCalc.fluid)
                s = PropsSI('S', 'T', T, 'P', p, InjectorCalc.fluid)
            except ValueError as e:
                print("State is in two-phase region. Please provide quality")
        else:
            h = PropsSI('H', 'T', T, 'Q', X, InjectorCalc.fluid)
            s = PropsSI('S', 'T', T, 'Q', X, InjectorCalc.fluid)

        print("Temperature = ", T, " [K]")
        print("Pressure = ", p/1000.0, " [kPa]")
        print("Enthalpy = ", h/1000.0, " [kJ/kg]")
        print("Entropy = ", s/1000.0, " [kJ/kgK]")

       
        
        
    
    def mSPI(states:list, N:int, d:float)->float:
        T1 = states[0]
        p1 = states[1]
        T2 = states[2]
        p2 = states[3]
        X2 = states[4]

        rho = PropsSI('D', 'T', T1, 'P', p1, InjectorCalc.fluid) #Density of inlet state

        A = InjectorCalc.holeArea(d)
        Cd:float = InjectorCalc.Cd(d)
        m = Cd*N*A*math.sqrt(2*rho*(p1-p2))
        
        return m

    def mHEM(states:list, N:int, d:float)->float:
        T1 = states[0]
        p1 = states[1]
        T2 = states[2]
        p2 = states[3]
        X2 = states[4]

        A = InjectorCalc.holeArea(d)
        Cd = InjectorCalc.Cd(d)

        if (X2 > 0) & (X2 < 1): #If saturated mixture, use quality
            
            rhof = PropsSI('D', 'T', T2, 'Q', 0, InjectorCalc.fluid)
            rhog = PropsSI('D', 'T', T2, 'Q', 1, InjectorCalc.fluid)

            rho =  (1-X2)*rhof + X2*rhog      #Use desnity of outlet state in HEM 
        else:
            rho = PropsSI('D', 'T', T2, 'P', p2, InjectorCalc.fluid)
        
        h1 = PropsSI('H', 'T', T1, 'P', p1, InjectorCalc.fluid)

        if (X2 > 0) & (X2 < 1): #If saturated mixture, use quality
            h2 = PropsSI('H', 'T', T2, 'Q', X2, InjectorCalc.fluid)
        else:
            h2 = PropsSI('H', 'T', T2, 'P', p2, InjectorCalc.fluid)
        
        m: float = Cd*N*A*rho*math.sqrt(2*(h1-h2))
        
        return m

    def mDyer(states:list, N:int, d:float)->float:
        T1 = states[0]
        p1 = states[1]
        T2 = states[2]
        p2 = states[3]
        X2 = states[4]
        
        try:
            psat = PropsSI('P', 'T', T1, 'Q', 0, InjectorCalc.fluid)
        except ValueError as e:
            print("Initial State is Supercritical Vapour - Dyer Model not Valid")
            sys.exit()
        try:
            k = math.sqrt((p1-p2)/(psat-p2))
        except ValueError as e:
            print("Initial Temperature Too Cold")

        mSPI = InjectorCalc.mSPI(states, N, d)
        mHEM = InjectorCalc.mHEM(states, N, d)

        return k/(1+k) * mSPI + 1/(1+k) * mHEM
    
    def getMassFlow(T1:float, p1:float, p2:float, N:int, d:float)->float:
        
        states = InjectorCalc.getExitState(T1, p1, p2)
        mDyer = InjectorCalc.mDyer(states, N, d)
        return round(mDyer, 2)
    

    #Secondary(Helper) Functions
    def Cd(d:float)->float: #enter diameter in mm 
         return 0.82291*math.exp(-0.0824635*d) #Formula based on exponential regression of Cd data vs Diameter in paper referenced above. Feel free to change if better methodology is found
    def holeArea(d:float)->float: #enter diameter in mm, returns area in m
        d/= 1000.0
        return 0.25*math.pi*d**2
    def C2K(TC:float)->float:
        return TC + 273.15
    def K2C(TK:float)->float:
        return round(TK - 273.15)

    
def main():

    os.system('cls')

    #System Parameters
    T1 = InjectorCalc.C2K(20)
    p1 = 5.5e6 #Takes in initial pressure in Pa 
    p2 = 101e3 #Takes in Combustion chamber pressure in Pa 
    
    #Injector Design Parameters
    N:int = 8 #Number of holes
    d = 1.4 #Diameter of holes in mm
    
    m = InjectorCalc.getMassFlow(T1, p1, p2, N, d)
        
    print(m, " [kg/s]")

    

if __name__ == "__main__":
    main() 