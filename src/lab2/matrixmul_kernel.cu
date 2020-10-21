/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	const int TILE_WIDTH = 32;
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	float Pvalue = 0.0;

	__shared__ float Ms[TILE_WIDTH*TILE_WIDTH], Ns[TILE_WIDTH*TILE_WIDTH];
	
	for (int i = 0; i < M.width/TILE_WIDTH + 1; i++){
		// Zero the paddings
		Ms[threadIdx.y*TILE_WIDTH + threadIdx.x] = 0.0;
		Ns[threadIdx.y*TILE_WIDTH + threadIdx.x] = 0.0;

		// Copy from global to share memory
		if (Row < M.height && i*TILE_WIDTH + threadIdx.x < M.width){
			Ms[threadIdx.y*TILE_WIDTH + threadIdx.x] = M.elements[Row*M.width + i*TILE_WIDTH + threadIdx.x];
		} 

		if (Col < N.width && i*TILE_WIDTH + threadIdx.y < N.height){
			Ns[threadIdx.y*TILE_WIDTH + threadIdx.x] = N.elements[(i*TILE_WIDTH + threadIdx.y)*N.width + Col];
		}
		__syncthreads();

		// Compute on shared memory
		for (int k = 0; k < TILE_WIDTH; k++){
			float M_elem = Ms[threadIdx.y*TILE_WIDTH + k];
			float N_elem = Ns[k*TILE_WIDTH + threadIdx.x];
			Pvalue += M_elem * N_elem;
		}
		__syncthreads();
	}
	// Copy back to global memory
	if (Row < P.height && Col < P.width){
		P.elements[Row*P.width + Col] = Pvalue;
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
