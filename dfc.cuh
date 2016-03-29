/**
* @file dfc.cuh
* @brief DFC code
* @author John Melton, G0ORX/N6LYT
*/


/* Copyright (C)
* 2015 - John Melton, G0ORX/N6LYT
*
* Based on code by Steven Passe AD0ES and Vasiliy Gokoyev K3IT
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

//#define DFC_FFT_INPUT_SIZE 1048576
//#define DFC_FFT_OVERLAP_SIZE 262144
//#define DFC_FFT_OUTPUT_SIZE ((DFC_FFT_INPUT_SIZE / 2) + 1)

extern cufftComplex* frequencysamples;
extern cufftComplex* deviceFrequencysamples;

extern sem_t frequencyBufferFull;
extern sem_t frequencyBufferEmpty;

void initDfc(void);
void dfcTerminate(void);

