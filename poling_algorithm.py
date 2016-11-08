#!/usr/bin/env python2.7

########################################################################################################
########################################################################################################
##  Copyright (c) 2016 Jean-Luc Tambasco                                                              ##
##                                                                                                    ##
##  Permission is hereby granted, free of charge, to any person obtaining a copy of this software     ##
##  and associated documentation files (the "Software"), to deal in the Software without restriction, ##
##  including without limitation the rights to use, copy, modify, merge, publish, distribute,         ##
##  sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is     ##
##  furnished to do so, subject to the following conditions:                                          ##
##                                                                                                    ##
##  The above copyright notice and this permission notice shall be included in all copies or          ##
##  substantial portions of the Software.                                                             ##
##                                                                                                    ##
##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING     ##
##  BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND        ##
##  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,      ##
##  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,    ##
##  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.           ##
########################################################################################################
########################################################################################################

import numpy as np
from scipy.integrate import cumtrapz

def SampledPoling(numDomainsPerPeriod, numPeriods, length, Chi2Profile,
                  filename=None, polingSineAmplitudeVisual=False,
                  leastSquaredError=True):
    # Even number of domains in QPM period.
    assert(numDomainsPerPeriod & 1 == 0)

    zPeriod = length/numPeriods
    zStep = zPeriod/2.
    z = np.arange(0., length+zStep, zStep)

    # Initial DC component of the SH is 0.
    dcESh = 0.
    # Maximum DC Esh that can be reached with ordinary QPM.
    maxDcESh = numPeriods
    # Assuming the amplitude of the Esh un-QPM oscillation is normalised to 1.Periods
    dcEShStep = 2.

    chi2Profile = Chi2Profile(z)

    # Generate SH E-field profile.
    shProfile = cumtrapz(chi2Profile, z)
    # Normalise assuming E-field at the end of ordinary QPM would be 1.
    shProfile /= length
    # Scale the maximum value to assume that a domain flipping
    # increases the second harmonic field DC component by 2.
    shProfile *= maxDcESh*2.
    if filename:
        np.savetxt(filename+'Chi2.dat', chi2Profile)
        np.savetxt(filename+'Efield.dat', shProfile)

    # *2 because two domains per period.
    poling = np.ones(numPeriods*2)
    pole = True
    dcWasFlipped1 = 0
    dcWasFlipped2 = 0

    shift1 = 1.
    shift2 = 1.
    wasFlipped1 = False
    wasFlipped2 = False

    if leastSquaredError:
        power = 2.
    else:
        power = 1.

    if polingSineAmplitudeVisual:
        sine = np.array([])
    for i in range(0, len(poling), 2):
        if i == len(shProfile)-1:
            e = shProfile[i] - dcESh
        elif i == len(shProfile)-2:
            e = shProfile[i+1] - dcESh
        else:
            e = shProfile[i+2] - dcESh
        error = np.sign(e)*e**power

        if error < -2:
            poling[i]   =  1
            if i != len(poling)-1:
                poling[i+1] = -1
            dcESh -= dcEShStep
        elif -2. < error < 2.:
            poling[i]   = 1
            if i != len(poling)-1:
                poling[i+1] = 1
        elif 2. < error:
            poling[i]   = -1
            if i != len(poling)-1:
                poling[i+1] =  1
            dcESh += dcEShStep

        if polingSineAmplitudeVisual and i <= len(shProfile)-3:
            if poling[i] == -1 and poling[i+1] == 1:
                shift1 *= -1
                dcWasFlipped1 += dcEShStep/2.
                dcWasFlipped2 += dcEShStep
            elif poling[i] == 1 and poling[i+1] == -1:
                shift2 *= -1
                dcWasFlipped2 -= dcEShStep/2.

            z = np.linspace(i, i+2, 100, False)
            sine = np.concatenate((sine, 0.5*shift1*np.cos(np.pi*z[:int(len(z)/2)])+dcWasFlipped1))
            sine = np.concatenate((sine, 0.5*shift2*np.cos(np.pi*z[int(len(z)/2):])+dcWasFlipped2))

            if poling[i] == -1 and poling[i+1] == 1:
                shift1 *= -1
                dcWasFlipped1 += dcEShStep/2.
            elif poling[i] == 1 and poling[i+1] == -1:
                shift2 *= -1
                dcWasFlipped1 -= dcEShStep
                dcWasFlipped2 -= dcEShStep/2.

    # Repeat to get correct number of domains per period.
    poling = np.repeat(poling, numDomainsPerPeriod/2)

    if filename:
        if polingSineAmplitudeVisual:
            np.savetxt(filename+'.dat', poling, '%.i')
            np.savetxt(filename+'Visual'+'.dat', sine, '%.4e')
        else:
            np.savetxt(filename+'.dat', poling, '%.i')

    return poling

def main():
    sd = 2.
    L  = 10.
    GausZ = lambda z: np.exp(-(z-L/2.)**2/(2.*sd**2))
    poling = SampledPoling(8, 50, L, GausZ, 'test', True)

if __name__ == '__main__':
    main()
