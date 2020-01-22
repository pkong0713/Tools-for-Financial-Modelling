"""
Simple Bond Valuation Calculator for Vanila Bonds using Present Value method
Model assumes flat term structure, which means that r = ytm = constant
Bond is created as a class, where users may input parameters and run different methods
Note that the compounding convention here is continous

Inputs: <Coupon Rate>, <Maturity>, <Face Value>, <Yield to Maturity>, <Number of Payments per Year>
Methds: self.get_value() to get valuation"""

import numpy as np
import pandas as pd
import math


class bond:
    def __init__(self,Coupon_Rate, Maturity, Face_Value, Yield_to_Maturity, n_Payment_per_year):
        """
        Define c, T, FV, r and n for Bond.
        Yield to Maturity must be in decimals."""

        self.df = pd.DataFrame(columns=['Time (years)','Interest Rate (Zero Rates)', 'Discount Factor', 'CF', 'DCF', 'Payment Number'])

        self.m = Maturity
        self.n = n_Payment_per_year
        self.r = Yield_to_Maturity #Flat term-structure, therefore the ytm
        self.fv = Face_Value
        self.N = self.m*self.n
        self.c = Coupon_Rate # Annualized
                
    def get_value(self):
        for payment_number in range(1,self.N):
            time = payment_number/self.n
            discount_factor = (math.exp(-self.r*time))
            C = self.fv*(self.c/self.n)
            dcf = C*discount_factor   
            self.df = self.df.append({'Time (years)': time, 'Interest Rate (Zero Rates)':self.r, 'Discount Factor': discount_factor, 'CF': C, 'DCF': dcf, 'Payment Number': payment_number}, ignore_index=True)
        
        discount_factor = (math.exp(-self.r*self.m))
        dcf = (C+self.fv)*discount_factor  
        df = self.df.append({'Time (years)': self.m, 'Interest Rate (Zero Rates)':self.r, 'Discount Factor': discount_factor, 'CF': C+self.fv, 'DCF': dcf, 'Payment Number': self.N}, ignore_index=True)
        
        B = int(df['DCF'].sum())
        print(B)

A = bond(0.12,3,100000,0.051541,2)

A.get_value()


""" Future Expansion:
    - self.is_discount()
    - clean price, dirty price and interest yield (IM6)
    - inclusion of term structure analysis (IM6)
    - different types of durations and convexivity (IM6 & DRM B.1)
    - bonds with embedded options"""