#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define fFloat double
#define ModeEnumCount 2

struct sInt
{
	const static int sIntDataSize = 4;
	unsigned int data[sIntDataSize];
	//Init as superzero
	sInt()
	{
		for (int i = 0; i < sIntDataSize; ++i) data[i] = 0x00000000;
	}

	__host__ __device__ sInt operator =(const sInt & a)
	{
		for (int i = 0; i < sIntDataSize; ++i) data[i] = a.data[i];
	}

	__host__ __device__ sInt operator +(const sInt & a)
	{
		for (int i = 0; i < sIntDataSize; ++i)
		{
			static bool overflow = false;
			unsigned int tmp = data[i] + a.data[i] + (overflow ? );
			overflow = tmp < data[i] || tmp < a.data[i];
		}
	}
};

typedef struct
{
	void * vboPointer;
	float time;
}SuperSineDataStruct;

typedef struct
{
	void * vboPointer;
	fFloat xmin, xmax;
	fFloat ymin, ymax;
	float3 colorLow, colorHigh;
	unsigned int max_iterations;
	fFloat edge;
	fFloat additionalNum1;
	fFloat additionalNum2;
}FractalDataStruct;

__global__ void SuperSineDraw(float time, void * buffer);
__global__ void PaethDestruction(char * pixelData);
__global__ void Glow(char * pixelData);

__global__ void MandelbrotFractalDraw(fFloat xmin, fFloat xmax, fFloat ymin, fFloat ymax, unsigned int max_iterations, fFloat edge, fFloat additionalNum1, fFloat additionalNum2, float3 colorLow, float3 colorHigh, char4 * pixels);

void sFractalCalculator(void * data);