/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _BICUBICTEXTURE_CU_
#define _BICUBICTEXTURE_CU_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>



#include <helper_math.h>

// includes, cuda
#include <helper_cuda.h>

typedef unsigned int uint;
typedef unsigned char uchar;

//#include "bicubicTexture_kernel.cuh"
#include "Ray.h"
#include "hitable.h"
#include "vec3.h"
#include "hitable_list.h"
#include "sphere.h"


extern "C" {
#include "Ray.h"
}

cudaArray *d_imageArray = 0;

extern "C" void initTexture(int imageWidth, int imageHeight, uchar *h_data)
{
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = d_imageArray;
}

extern "C" void freeTexture()
{
	checkCudaErrors(cudaFreeArray(d_imageArray));
}

__device__ static int ticks = 1;

__device__ vec3 castRay(const ray& r, hitable **world)
{	
	hit_record rec;
	if ((*world)->hit(r, 0.0, FLT_MAX, rec)) 
	{		 
		return 0.5f*vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
	}
	else 
	{
		vec3 unit_direction = unit_vector(r.direction());
		float t = 0.5f*(unit_direction.y() + 1.0f);
		return (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	}	
}

#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
__device__ const int particlecountDevice = 50 + 5;
const int particlecountHost = 50 + 5;
__device__ const float boundary = 50; // +/- x and y boundaries
hitable **d_list;
hitable **d_world;
int posXrand[particlecountHost] = {};
int posYrand[particlecountHost] = {};
int posZrand[particlecountHost] = {};
int velXrand[particlecountHost] = {};
int velYrand[particlecountHost] = {};
int velZrand[particlecountHost] = {};

__global__ void create_world(hitable **d_list, hitable **d_world, int *posX, int *posY, int *posZ, int *velX, int *velY, int *velZ) 
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		// Create borders (Walls etc. are just oversized spheres)
		d_list[0] = new sphere(vec3(0, 0, -10050), 10000, 0, 0, 0); // Rear wall
		d_list[1] = new sphere(vec3(-10050, 0, -3), 10000, 0, 0, 0); // Left wall
		d_list[2] = new sphere(vec3(10050, 0, -3), 10000, 0, 0, 0); // Right wall
		d_list[3] = new sphere(vec3(0, -10050, -1), 10000, 0, 0, 0); // Floor
		d_list[4] = new sphere(vec3(0, 10050, -1), 10000, 0, 0, 0); // Ceiling
		
		// Initialise spheres into empty space		
		for (int i = 5; i < particlecountDevice; i++) {
			if (velX[i] == 0 && velY[i] == 0 && velZ[i] == 0) {
				velX[i] += 1;
			}
			d_list[i] = new sphere(vec3(posX[i], posY[i], posZ[i]), 0.5, velX[i], velY[i], velZ[i]);
		}

		*d_world = new hitable_list(d_list, particlecountDevice);
	}
}

__global__ void free_world(hitable **d_list, hitable **d_world)
{
	for (int i = 0; i < particlecountDevice; i++) {
		delete d_list[i];
	}
	delete *d_world;
}

__global__ void updatePositions(hitable **d_list)
{
	uint particleindlist = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Indexes 0-4 are the Walls, floor and ceiling.
	if (particleindlist < 5)
		return;
	
	sphere* kernelSphere = (sphere*)(d_list[particleindlist]);
	float x = kernelSphere->center.x();
	float y = kernelSphere->center.y();
	float z = kernelSphere->center.z();

	// Retrieve object velocities
	float vx = kernelSphere->_vx;
	float vy = kernelSphere->_vy;
	float vz = kernelSphere->_vz;

	float newX = x + vx;
	float newY = y + vy;
	float newZ = z + vz;

	if (newX > boundary) {
		newX = boundary - (newX - boundary);
		vx = -vx; // Reverse velocity on offending axis due to rebound
	}
	else if (newX < -boundary) {
		newX = newX - (newX - -boundary);
		vx = -vx;
	}
	else if (newX == boundary || newX == -boundary)
		vx = -vx;

	if (newY > boundary) {
		newY = boundary - (newY - boundary);
		vy = -vy; // Reverse velocity on offending axis due to rebound
	}
	else if (newY < -boundary) {
		newY = newY - (newY - -boundary);
		vy = -vy;
	}
	else if (newY == boundary || newY == -boundary)
		vy = -vy;

	if (newZ < -boundary) {
		newZ = -boundary - (newZ - -boundary);
		vz = -vz; // Reverse velocity on offending axis due to rebound
	}
	else if (newZ > 0) {
		newZ = -newZ;
		vz = -vz;
	}
	else if (newZ == boundary || newZ == 0)
		vz = -vz;

	// Update position and velocities
	kernelSphere->setPos(newX, newY, newZ);
	kernelSphere->setVel(vx, vy, vz);

	// Re-assign object at pointer location
	d_list[particleindlist] = kernelSphere;
}

__global__ void gravityFall(hitable **d_list)
{
	uint particleindlist = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleindlist < 5)
		return;

	sphere* kernelSphere = (sphere*)(d_list[particleindlist]);
	float x = kernelSphere->center.x();
	float y = kernelSphere->center.y();
	float z = kernelSphere->center.z();

	float newY = y - 1;
	
	if (newY < -boundary)
		newY = boundary;

	kernelSphere->setPos(x, newY, z);
	d_list[particleindlist] = kernelSphere;
}

__global__ void d_render(uchar4 *d_output, uint width, uint height, hitable **d_world)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint i = y * width + x;

	// Labs/Lectures code
	float u = x / (float)width; //----> [0, 1]x[0, 1]
	float v = y / (float)height;
	u = 2.0*u - 1.0; //---> [-1, 1]x[-1, 1]
	v = -(2.0*v - 1.0);
	u *= width / (float)height;
	u *= 2.0;
	v *= 2.0;
	vec3 eye = vec3(0, 0.5, 1.5);
	float distFrEye2Img = 1.0;;
	if ((x < width) && (y < height))
	{
		//for each pixel
		vec3 pixelPos = vec3(u, v, eye.z() - distFrEye2Img);
		//fire a ray:
		ray r;
		r.rayOri = eye;
		r.rayDir = pixelPos - eye; //view direction along negtive z-axis!
		vec3 col = castRay(r, d_world);

		// Colours
		float red = col.x();
		float green = col.y();
		float blue = col.z();

		d_output[i] = make_uchar4(red * 255, green * 255, blue * 255, 0);
	}
}

#include <time.h>
extern "C" void initialWorld() {
	// Generate random numbers
	// Random positions arrays
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		posXrand[i] = rand() % 100;
		posXrand[i] -= 50;
	}
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		posYrand[i] = rand() % 200 - 100;
		posYrand[i] -= 50;
		while (posYrand[i] < -50) {
			posYrand[i] += 50;
		}
	}
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		posZrand[i] = rand() % 50;
		posZrand[i] -= 50;
	}

	// Random velocities arrays
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		velXrand[i] = rand() % 6;
		velXrand[i] -= 3;
	}
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		velYrand[i] = rand() % 8 - 5;
		while (velYrand[i] < -3) {
			velYrand[i] += 3;
		}
	}
	srand(time(NULL));
	for (int i = 0; i < particlecountHost; ++i) {
		velZrand[i] = rand() % 7 - 4;
		while (velZrand[i] < -3) {
			velZrand[i] += 3;
		}
	}

	// Allocate and fill GPU memory
	int *xrand = 0;
	int *yrand = 0;
	int *zrand = 0;
	int *vxrand = 0;
	int *vyrand = 0;
	int *vzrand = 0;
	checkCudaErrors(cudaMalloc((void **)&xrand, particlecountHost * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&yrand, particlecountHost * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&zrand, particlecountHost * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&vxrand, particlecountHost * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&vyrand, particlecountHost * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&vzrand, particlecountHost * sizeof(int)));
	cudaMemcpy(xrand, posXrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(yrand, posYrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(zrand, posZrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vxrand, velXrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vyrand, velYrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(vzrand, velZrand, particlecountHost * sizeof(int), cudaMemcpyHostToDevice);		
	
	// Allocate GPU memory
	checkCudaErrors(cudaMalloc((void **)&d_list, particlecountHost * sizeof(hitable *)));
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *))); 

	create_world << <1, 1 >> > (d_list, d_world, xrand, yrand, zrand, vxrand, vyrand, vzrand);

	cudaFree(xrand);
	cudaFree(yrand);
	cudaFree(zrand);
	cudaFree(vxrand);
	cudaFree(vyrand);
	cudaFree(vzrand);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

// render image using CUDA
extern "C" void render(int width, int height, dim3 blockSize, dim3 gridSize, uchar4 *output)
{
	d_render <<<gridSize, blockSize>>> (output, width, height, d_world);
	getLastCudaError("kernel failed");
	//free_world << <1, 1 >> > (d_list, d_world);
}

extern "C" void positionsUpdate()
{
	updatePositions << <1, particlecountHost >> > (d_list); // 1 thread per particle
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void gravityEngaged() 
{
	gravityFall << <1, particlecountHost >> > (d_list); // 1 thread per particle
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) 
{
	if (result) 
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


#endif
