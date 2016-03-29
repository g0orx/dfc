/**
* @file coeff.cu
* @brief Filter coefficients
* @author John Melton, G0ORX/N6LYT
*/


/* Copyright (C)
* 2015 - John Melton, G0ORX/N6LYT
*
* Based on code from WDSP written by Warren Pratt, NR0V
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
*
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double Ino(double x) {
  /*
   * This function calculates the zeroth order Bessel function
   */
  double d = 0.0, ds = 1.0, s = 1.0;
  do
  {
    d += 2.0;
    ds *= x*x/(d*d);
    s += ds;
  }
  while (ds > s*1e-6);
  return s;
}

double *calcFilter(double Fs, double Fa, double Fb, int M, double Att) {

  int Np = (M-1)/2;
  double A[Np+1];
  double Alpha;
  int j;
  double pi = 3.1415926535897932;
  double Inoalpha;
  double *H;

  H=(double*)malloc(M*sizeof(double));
    // Calculate the impulse response of the ideal filter
    A[0] = 2.0*(Fb-Fa)/Fs;
    for(j=1; j<=Np; j++)
    {
      A[j] = (sin(2.0*(double)j*pi*Fb/Fs)-sin(2.0*(double)j*pi*Fa/Fs))/((double)j*pi);
    }
    // Calculate the desired shape factor for the Kaiser-Bessel window
    if (Att<21.0)
    {
      Alpha = 0.0;
    }
    else if (Att>50.0)
    {
      Alpha = 0.1102*(Att-8.7);
    }
    else
    {
      Alpha = 0.5842*pow((Att-21.0), 0.4)+0.07886*(Att-21.0);
    }
    // Window the ideal response with the Kaiser-Bessel window
    Inoalpha = Ino(Alpha);
    for (j=0; j<=Np; j++)
    {
      H[Np+j] = A[j]*Ino(Alpha*sqrt(1.0-((double)j*(double)j/((double)Np*(double)Np))))/Inoalpha;
    }
    for (j=0; j<Np; j++)
    {
      H[j] = H[M-1-j];
    }

    return H;
}

