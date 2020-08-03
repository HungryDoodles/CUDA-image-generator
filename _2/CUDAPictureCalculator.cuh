#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define fFloat double
#define ModeEnumCount 4
#define MaxRenderingTesselation 9
#define fdata progressiveFractalData.renderingData
#define pfdata progressiveFractalData
#define vfdata progressiveVolFractalData.renderingData
#define vpfdata progressiveVolFractalData
#define blockSize 128
#define MinRaymarcherDistance 0.001
#define NormalDelta 0.00001

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
	float colorExp;
}FractalDataStruct;

typedef struct
{
	uchar4 * pixels;
	double3 loc, dir;
	fFloat FOV;
	float3 colorLow, colorHigh;
	unsigned int max_iterations;
	unsigned int raySteps;
	bool stopOnHit;
	fFloat edge;
	fFloat minStep;
	fFloat stepMul;
	fFloat additionalNum1;
	fFloat additionalNum2;
	float brightness;
	float colorExp;
}VolFractalDataStruct;

typedef struct
{
	FractalDataStruct renderingData;
	void * frameBuffer;
	void * injectionBuffer;
	void * blockBuffer;
	unsigned int currentRenderingTesselation;
	unsigned int renderingBlockNum;
}progressiveFractalDataType;

typedef struct
{
	VolFractalDataStruct renderingData;
	void * frameBuffer;
	void * injectionBuffer;
	void * blockBuffer;
	unsigned int currentRenderingTesselation;
	unsigned int renderingBlockNum;
}progressiveVolFractalDataType;

enum Mode
{
	superSine = 0, fractal = 1, progressiveFractal = 2, volFractal = 3
};

__host__ __device__ void rotateVec(double3 * v, fFloat aroundZ, fFloat aroundY);
__host__ __device__ double3 headingVec(fFloat aroundZ, fFloat aroundY);
__host__ __device__ void normalizeVec(double3 * v);
__host__ __device__ double lenVec(double3 * v);
__host__ __device__ void scaleVec(double3 * v, double scale);
__host__ __device__ void addVec(double3 * v, double3 other);
__host__ __device__ double3 crossVec(double3 a, double3 b);
__host__ __device__ double dotVec(double3 a, double3 b);

__global__ void SuperSineDraw(float time, void * buffer);
__global__ void PaethDestruction(char * pixelData);
__global__ void Glow(char * pixelData);

__device__ double3 getDENorm(double3 point, double(*DE)(double3, double3));
__device__ double3 getDENorm(double3 point, double ap1, double ap2, double(*DE)(double3, double, double));
__device__ double MandelbulbDE(double3 pos, double additionParam1, double additionalParam2);
__device__ double HashmapDE(double3 pos, double additionalParam1, double additionalParam2);
__device__ double SphereDE(double3 pos, double3 offset);
__device__ double SphereCombinedDE(double3 pos, double3 offset);
__device__ double4 Raymarcher(double3 rayDir, double3 rayPos, int maxSteps, double additionalParam1, double additionalParam2);
__global__ void MandelbrotFractalDraw(fFloat xmin, fFloat xmax, fFloat ymin, fFloat ymax, unsigned int max_iterations, fFloat edge, fFloat additionalNum1, fFloat additionalNum2, float3 colorLow, float3 colorHigh, float colorExp, uchar4 * pixels);
__global__ void MandelbrotFractalDrawPrepass(fFloat xmin, fFloat xmax, fFloat ymin, fFloat ymax, unsigned int max_iterations, fFloat edge, fFloat additionalNum1, fFloat additionalNum2, float3 colorLow, float3 colorHigh, float colorExp, uchar4 * pixels);
__global__ void VolMandelbrotFractalDraw(fFloat xmin, fFloat xmax, fFloat ymin, fFloat ymax, VolFractalDataStruct data, uchar4 * pixels);

__global__ void ClearBuffer(char4 * pbo);
__global__ void SupersampleOnce(int factor, uchar4 * from, uchar4 * to);
__global__ void SubsampleSimple(int factor, char4 * from, char4 * to);
__global__ void Inject(int mainFactor, int sideFactor, int blockShiftX, int blockShiftY, char4 * from, char4 * to);
__global__ void Render(int stretchFactor, char4 * from, char4 * to);

void PictureCalculator(Mode mode, void * data);

void DrawProgressiveFractal(FractalDataStruct * fractalData);
void DrawProgressiveVolFractal(VolFractalDataStruct * fractalData);

void fractalInit();