#include "CUDAPictureCalculator.cuh"
#include "cuComplex.h"
#include "math.h"

void sFractalCalculator(void * data)
{
	fractalData = *((FractalDataStruct*)data);
	MandelbrotFractalDraw<<<1024, 1024>>>(
		fractalData.xmin, fractalData.xmax,
		fractalData.ymin, fractalData.ymax,
		fractalData.max_iterations,
		fractalData.edge,
		fractalData.additionalNum1,
		fractalData.additionalNum2,
		fractalData.colorLow, fractalData.colorHigh,
		(char4 *)fractalData.vboPointer);
	cudaDeviceSynchronize();
	break;
}

//SINE
__global__ void SuperSineDraw(float time, void * buffer)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	char * currentByte = (char *)buffer;
	currentByte += 4*(x + (y << 10));

	const float3 baseColor = { 0.05f, 0.05f, 1.0f };
	const float BrightnessMultiplier = 30.0f;

	float pixelValue = 0;
	pixelValue = sinf(((float)x) / 32 + (time) / 2);
	pixelValue += sinf(((float)x) / 17 + (time) / 1.7f) * 0.6f;
	pixelValue += sinf(((float)x) / 67 + (time) / 1.33f) * 0.4f;
	pixelValue *= 0.3f;
	//pixelValue *= multiplier;

	pixelValue = expf(-abs((pixelValue * 256 - 512 + y) * 0.02f));

	*(currentByte++) = 255-(char)((tanhf(pixelValue * baseColor.x * BrightnessMultiplier)) * 255);
	*(currentByte++) = 255-(char)((tanhf(pixelValue * baseColor.y * BrightnessMultiplier)) * 255);
	*(currentByte++) = 255-(char)((tanhf(pixelValue * baseColor.z * BrightnessMultiplier)) * 255);
	/**(currentByte++) = 255;
	*(currentByte++) = 1;
	*(currentByte++) = 8;*/
}

__global__ void PaethDestruction(char * pixelData)
{
	for (int i = 0; i < 3; ++i)
	{
		int n = threadIdx.x * 4 + i;
		signed short left = pixelData[n + (blockDim.x + 1) * 4 * (blockIdx.x + 1)];
		signed short top = pixelData[n + 4 + (blockDim.x + 1) * 4 * blockIdx.x];
		signed short upperLeft = pixelData[n + (blockDim.x + 1) * 4 * blockIdx.x];
		//pixelData[n + 1 + (blockDim.x + 1) * 4 * (blockIdx.x + 1)] -= (char)((left + top + upperLeft) / 3);
		signed short p = left + top - upperLeft;
		signed short pa = abs(p - left);
		signed short pb = abs(p - top);
		signed short pc = abs(p - upperLeft);
		if (pa >= pb && pa >= pc) pixelData[n + 4 + (blockDim.x+1) * 4 * (blockIdx.x+1)] = (char)pa;
		else if (pb >= pc) pixelData[n + 4 + (blockDim.x+1) * 4 * (blockIdx.x+1)] = (char)pb;
		else pixelData[n + 4 + (blockDim.x+1) * 4 * (blockIdx.x+1)] = (char)pc;
	}
}

__global__ void Glow(char * pixelData)
{
	int x = threadIdx.x + 1;
	int y = blockIdx.x + 1;

	char * currentByte = pixelData + (x + (y << 10)) * 4;
	float fGrey = (float)(*currentByte) / 255;

	const float expMinusOneMul = 0.36788f;
	const float expMinusSqrtTwoMul = 0.24312f;
	//const float expMinusOneMul = 0;
	//const float expMinusSqrtTwoMul = 0;
	for (int i = 0; i < 3; ++i)
	{
		//*(currentByte) = (char)(tanhf(((float)(*(currentByte))*2)) * 255);

		*(currentByte + 4096) = (char)(tanhf(((float)(*(currentByte + 4096)) / 255 + fGrey * expMinusOneMul)) * 255);
		*(currentByte - 4096) = (char)(tanhf(((float)(*(currentByte - 4096)) / 255 + fGrey * expMinusOneMul)) * 255);
		*(currentByte + 4) = (char)(tanhf(((float)(*(currentByte +    4)) / 255 + fGrey * expMinusOneMul)) * 255);
		*(currentByte - 4) = (char)(tanhf(((float)(*(currentByte -    4)) / 255 + fGrey * expMinusOneMul)) * 255);

		*(currentByte + 4100) = (char)(tanhf(((float)(*(currentByte + 4100)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte - 4100) = (char)(tanhf(((float)(*(currentByte - 4100)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte + 4092) = (char)(tanhf(((float)(*(currentByte + 4092)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte - 4092) = (char)(tanhf(((float)(*(currentByte - 4092)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);

		++currentByte;
	}
}
//Fractals
__global__ void MandelbrotFractalDraw(
	fFloat xmin, fFloat xmax,
	fFloat ymin, fFloat ymax,
	unsigned int max_iterations,
	fFloat edge,
	fFloat additionalNum1, fFloat additionalNum2,
	float3 colorLow, float3 colorHigh,
	char4 * pixels)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	unsigned int iterations = 0;
	/*
	float cx = xmin + x / 1023.0*(xmax - xmin);
	float cy = ymin + y / 1023.0*(ymax - ymin);

	for (iterations = 0; iterations < max_iterations && (cy * cy + cx * cx) < edge; ++iterations)
	{
	cy = cos((sin(cx * cy) + log(additionalNum1 * cx)) + additionalNum2);
	}
	*/
	double2 c = { additionalNum1, additionalNum2 };
	double2 z = { xmin + x / 1023.0*(xmax - xmin), ymin + y / 1023.0*(ymax - ymin) };

	double2 tmpz;

	for (iterations = 0; iterations < max_iterations && (z.x * z.x + z.y * z.y) < edge; ++iterations)
	{
		tmpz.x = z.x * z.x - z.y * z.y + c.x;
		tmpz.y = 2 * z.x * z.y + c.y;
		z.x = tmpz.x;
		z.y = tmpz.y;
	}
	//z = cuCadd(cuCmul(z, z), c);

	char4 * currentPixel = pixels + (x + y * 1024);

	float3 resultColor;
	if (iterations != max_iterations)
	{
		float coeff = (float)(max_iterations - iterations) / max_iterations;

		resultColor =
		{
			colorLow.x * (1 - coeff) + colorHigh.x * coeff,
			colorLow.y * (1 - coeff) + colorHigh.y * coeff,
			colorLow.z * (1 - coeff) + colorHigh.z * coeff
		};
	}
	else
	{
		resultColor = { 0, 0, 0 };
	}

	*(currentPixel) =
	{
		resultColor.x * 255,
		resultColor.y * 255,
		resultColor.z * 255,
		255
	};
}