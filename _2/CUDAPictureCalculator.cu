#include "CUDAPictureCalculator.cuh"
#include "cuComplex.h"
#include <math.h>
//#include "math_functions.h"
#include <iostream>

progressiveFractalDataType progressiveFractalData;
progressiveVolFractalDataType progressiveVolFractalData;

void fractalInit()
{
	progressiveFractalData.frameBuffer = nullptr;
	progressiveFractalData.injectionBuffer = nullptr;
	progressiveFractalData.blockBuffer = nullptr;
	progressiveFractalData.currentRenderingTesselation = 1;
	progressiveFractalData.renderingBlockNum = 0;
	progressiveVolFractalData.frameBuffer = nullptr;
	progressiveVolFractalData.injectionBuffer = nullptr;
	progressiveVolFractalData.blockBuffer = nullptr;
	progressiveVolFractalData.currentRenderingTesselation = 1;
	progressiveVolFractalData.renderingBlockNum = 0;
}

bool isDataChanged(FractalDataStruct * data)
{
	if (data)
		return
		data->xmin != fdata.xmin ||
		data->xmax != fdata.xmax ||
		data->ymin != fdata.ymin ||
		data->ymax != fdata.ymax ||
		data->max_iterations != fdata.max_iterations ||
		data->edge != fdata.edge ||
		data->additionalNum1 != fdata.additionalNum1 ||
		data->additionalNum2 != fdata.additionalNum2 ||
		data->colorExp != fdata.colorExp;
	else return true;
}
bool double3Equal(double3 left, double3 right)
{
	return
		left.x == right.x &&
		left.y == right.y &&
		left.z == right.z;
}
bool isDataChanged(VolFractalDataStruct * data)
{
	if (data)
		return
		!double3Equal(data->loc, vfdata.loc) ||
		!double3Equal(data->dir, vfdata.dir) ||
		data->FOV != vfdata.FOV ||
		data->stopOnHit != vfdata.stopOnHit ||
		data->max_iterations != vfdata.max_iterations ||
		data->edge != vfdata.edge ||
		data->additionalNum1 != vfdata.additionalNum1 ||
		data->additionalNum2 != vfdata.additionalNum2 ||
		data->colorExp != vfdata.colorExp ||
		data->brightness != vfdata.brightness;
	else return true;
}

void PictureCalculator(Mode mode, void * data)
{
	FractalDataStruct fractalData = *((FractalDataStruct*)data);
	switch (mode)
	{
	case superSine:
		SuperSineDraw<<<1024, 1024 >>>((*((SuperSineDataStruct*)data)).time, 
										(*((SuperSineDataStruct*)data)).vboPointer);
		PaethDestruction <<<1023, 1023 >>>((char*)(*((SuperSineDataStruct*)data)).vboPointer);
		PaethDestruction <<<1023, 1023 >>>((char*)(*((SuperSineDataStruct*)data)).vboPointer);
		Glow<<<1022, 1022>>>((char*)(*((SuperSineDataStruct*)data)).vboPointer);
		cudaDeviceSynchronize();
		break;
	case fractal:
		MandelbrotFractalDraw<<<1024,1024>>>(
			fractalData.xmin, fractalData.xmax,
			fractalData.ymin, fractalData.ymax,
			fractalData.max_iterations,
			fractalData.edge,
			fractalData.additionalNum1,
			fractalData.additionalNum2,
			fractalData.colorLow, fractalData.colorHigh, fractalData.colorExp,
			(uchar4 *)fractalData.vboPointer);
		cudaDeviceSynchronize();
		break;
	case progressiveFractal:
		DrawProgressiveFractal(&fractalData);
		cudaDeviceSynchronize();
		break;
	case volFractal:
		DrawProgressiveVolFractal((VolFractalDataStruct*)data);
		cudaDeviceSynchronize();
		break;
	}
}

//GEOMETRY
__host__ __device__ void rotateVec(double3 * v, fFloat aroundZ, fFloat aroundY)
{
	fFloat sinz = sin(aroundZ);
	fFloat cosz = cos(aroundZ);
	fFloat siny = sin(aroundY);
	fFloat cosy = cos(aroundY);
	//Y
	v->x = v->x * cosy + v->z * siny;
	v->z = -v->x *siny + v->z * cosy;
	//Z
	v->x = v->x * cosz - v->y * sinz;
	v->y = v->x * sinz + v->y * cosz;
}
__host__ __device__ double3 headingVec(fFloat aroundZ, fFloat aroundY)
{
	return
	{
		cos(aroundZ) * cos(aroundY),
		sin(aroundZ) * cos(aroundY),
		sin(aroundY)
	};
}
__host__ __device__ void normalizeVec(double3 * v)
{
	double len = sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
	if (len != 0)
	{
		v->x /= len;
		v->y /= len;
		v->z /= len;
	}
}
__host__ __device__ void scaleVec(double3 * v, double scale)
{
	v->x *= scale;
	v->y *= scale;
	v->z *= scale;
}
__host__ __device__ void addVec(double3 * v, double3 other)
{
	v->x += other.x;
	v->y += other.y;
	v->z += other.z;
}
__host__ __device__ double lenVec(double3 * v)
{
	return sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
}
__host__ __device__ double3 crossVec(double3 a, double3 b)
{
	return
	{
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	};
}
__host__ __device__ double dotVec(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
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
		else if(pb >= pc) pixelData[n + 4 + (blockDim.x+1) * 4 * (blockIdx.x+1)] = (char)pb;
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
		*(currentByte + 4) =    (char)(tanhf(((float)(*(currentByte +    4)) / 255 + fGrey * expMinusOneMul)) * 255);
		*(currentByte - 4) =    (char)(tanhf(((float)(*(currentByte -    4)) / 255 + fGrey * expMinusOneMul)) * 255);

		*(currentByte + 4100) = (char)(tanhf(((float)(*(currentByte + 4100)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte - 4100) = (char)(tanhf(((float)(*(currentByte - 4100)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte + 4092) = (char)(tanhf(((float)(*(currentByte + 4092)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);
		*(currentByte - 4092) = (char)(tanhf(((float)(*(currentByte - 4092)) / 255 + fGrey * expMinusSqrtTwoMul)) * 255);

		++currentByte;
	}
}
//Raymarcher
__device__ double3 getDENorm(double3 p, double(*DE)(double3, double3))
{
	double3 normal = 
	{
		DE({ p.x + NormalDelta, p.y, p.z }, { 0, 0, 0 }) - DE({ p.x - NormalDelta, p.y, p.z }, { 0, 0, 0 }),
		DE({ p.x, p.y + NormalDelta, p.z }, { 0, 0, 0 }) - DE({ p.x, p.y - NormalDelta, p.z }, { 0, 0, 0 }),
		DE({ p.x, p.y, p.z + NormalDelta }, { 0, 0, 0 }) - DE({ p.x, p.y, p.z - NormalDelta }, { 0, 0, 0 })
	};
	normalizeVec(&normal);
	return normal;
}
__device__ double3 getDENorm(double3 p, double ap1, double ap2, double(*DE)(double3, double, double))
{
	double3 normal =
	{
		DE({ p.x + NormalDelta, p.y, p.z }, ap1, ap2) - DE({ p.x - NormalDelta, p.y, p.z }, ap1, ap2),
		DE({ p.x, p.y + NormalDelta, p.z }, ap1, ap2) - DE({ p.x, p.y - NormalDelta, p.z }, ap1, ap2),
		DE({ p.x, p.y, p.z + NormalDelta }, ap1, ap2) - DE({ p.x, p.y, p.z - NormalDelta }, ap1, ap2)
	};
	normalizeVec(&normal);
	return normal;
}
__device__ double MandelbulbDE(double3 pos, double additionParam1, double additionalParam2)
{
	double3 z = pos;
	double dr = 1.0;
	double r = 0.0;
	for (int i = 0; i < 250; i++) {
		r = lenVec(&z);
		if (r>4) break;

		// convert to polar coordinates
		double theta = acos(z.z / r);
		double phi = atan2(z.y, z.x);
		dr = pow(r, additionParam1 - 1.0) * additionParam1 * dr + 1.0;

		// scale and rotate the point
		double zr = pow(r, additionParam1);
		theta = theta*additionParam1;
		phi = phi * additionParam1;

		// convert back to cartesian coordinates
		z = { zr*sin(theta)*cos(phi), zr*sin(phi)*sin(theta), zr*cos(theta) };
		addVec(&z, pos);
	}
	return 0.5*log(r)*r / dr;
}
__device__ double HashmapDE(double3 pos, double additionalParam1, double additionalParam2)
{
	/*vec4 hash(vec3 coord)
{
	//    gridcell is assumed to be an integer coordinate
	const vec3 OFFSET = vec3(26.0, 161.0, 2166.0);
	const float DOMAIN = 71.0;
	const float SOMELARGEFLOAT = 951.135664;
	vec4 P = vec4(coord.xyz, length(coord.xyz) + 1.0);
	P = P - floor(P * (1.0 / DOMAIN)) * DOMAIN;    //    truncate the domain
	P += OFFSET.xyxy;                                //    offset to interesting part of the noise
	P *= P;                                          //    calculate and return the hash
	return fract(P.xzxz * P.yyww * (1.0 / SOMELARGEFLOAT.x).xxxx);
}*/
	//Roundup
	pos = { (long)(pos.x), (long)(pos.y), (long)(pos.z) };
	double poslen = lenVec(&pos) + 1.0;

	//double3 offset = { 26.0, 161.0, 2166.0 };
	double3 offset = { 0, 0, 0 }; // What if..
	double domain = 71.0 * additionalParam1;
	double largeFloat = 951.135664 * additionalParam2;
	double4 P = { pos.x, pos.y, pos.z, poslen };
	P.x = P.x - floor(P.x * (1.0 / domain)) * domain;
	P.y = P.y - floor(P.y * (1.0 / domain)) * domain;
	P.z = P.z - floor(P.z * (1.0 / domain)) * domain;
	P.w = P.w - floor(P.w * (1.0 / domain)) * domain;

	P = { P.x + offset.x, P.y + offset.y, P.z + offset.z, P.w + offset.x };
	P = { P.x * P.x, P.y * P.y, P.z * P.z, P.w * P.w };
	double largeFloatCalc = (1.0 / largeFloat);
	P = { P.x * P.y * largeFloatCalc, P.z * P.y * largeFloatCalc, P.x * P.w * largeFloatCalc, P.z * P.w * largeFloatCalc };
	P = { P.x - (long)P.x, P.y - (long)P.y, P.z - (long)P.z, P.w - (long)P.w };
	
	double pointLen = sqrt(P.x * P.x + P.y * P.y + P.z * P.z + P.w * P.w); // 4D
	//pointLen *= 1600.f;
	//pointLen = pow(pointLen, additionalParam1);
	//pointLen *= additionalParam2;
	return pointLen;
}
__device__ double SphereDE(double3 pos, double3 offset)
{
	addVec(&pos, offset);
	return fmax(0.0, lenVec(&pos) - 1.0);
}
__device__ double SphereCombinedDE(double3 pos, double3 offset)
{
	addVec(&pos, offset);
	double3 offPos = pos;
	addVec(&offPos, { 0, 2, 0 });
	return fmin(lenVec(&pos) - 1.0, lenVec(&offPos) - 1.7);
}
__device__ double4 Raymarcher(double3 rayDir, double3 rayPos, int maxSteps, double additionalParam1, double additionalParam2)
{
	double traveledDistance = 0;
	double minMeasured = 100000000;//Just a lot
	int steps;
	double3 p = rayPos;
	for (steps = 0; steps < maxSteps; ++steps)
	{
		// p = rayPos + rayDir * traveledDistance;
		//double3 dir = rayDir;
		//scaleVec(&dir, traveledDistance);
		//addVec(&p, dir);

		//double distance = MandelbulbDE(p, additionalParam1, additionalParam2);
		double distance = HashmapDE(p, additionalParam1, additionalParam2);
		if (minMeasured > distance) minMeasured = distance;
		traveledDistance += distance;

		double3 dir = rayDir;
		double3 shiftVec = rayPos; float scale = lenVec(&shiftVec);
		scaleVec(&shiftVec, -1 * distance * additionalParam2);
		addVec(&dir, shiftVec);
		normalizeVec(&dir);
		scaleVec(&dir, distance);
		addVec(&p, dir);

		if (MinRaymarcherDistance > distance)
			break;
	}
	double3 endVec = p;
	//scaleVec(&endVec, traveledDistance);
	//addVec(&endVec, rayPos);
	if(steps < maxSteps) return{ endVec.x, endVec.y, endVec.z, 1.0 - (double)steps / maxSteps };
	return{ endVec.x, endVec.y, endVec.z, -minMeasured/2.0 }; // 2.0 is max magnitude of hashmap
}


__global__ void MandelbrotFractalDrawPrepass(
	fFloat xmin, fFloat xmax, 
	fFloat ymin, fFloat ymax, 
	unsigned int max_iterations, 
	fFloat edge,
	fFloat additionalNum1, fFloat additionalNum2, 
	float3 colorLow, float3 colorHigh, float colorExp, 
	uchar4 * pixels)
{

}

//Fractals
__global__ void MandelbrotFractalDraw(
	fFloat xmin, fFloat xmax,
	fFloat ymin, fFloat ymax,
	unsigned int max_iterations,
	fFloat edge,
	fFloat additionalNum1, fFloat additionalNum2,
	float3 colorLow, float3 colorHigh,  float colorExp,
	uchar4 * pixels)
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

	// COMPLEX MANDELBROT

	double2 z = {additionalNum1, additionalNum2};
	double xOffs = xmin + x / (double)(blockDim.x - 1)*(xmax - xmin);
	double yOffs = ymin + y / (double)(blockDim.x - 1)*(ymax - ymin);
	//double radius = sqrt(abs(xOffs * xOffs + yOffs * yOffs));
	//double angle = atan2(yOffs + additionalNum1, xOffs + additionalNum2);
	double2 c = { xOffs, yOffs };

	double2 tmpz;

	//HYPERCOMPLEX MANDELBROT
	/*double4 z = { 0, 0, 0, 0 };
	double xOffs = xmin + x / (double)(blockDim.x - 1)*(xmax - xmin);
	double yOffs = ymin + y / (double)(blockDim.x - 1)*(ymax - ymin);
	double4 c = { xOffs, yOffs, additionalNum1, additionalNum2 };
	double4 t;

	for (iterations = 0; iterations < max_iterations && (z.x * z.x + z.y * z.y + z.z * z.z + z.w * z.w) < edge; ++iterations)
	{
		t.x = (z.x * z.x - z.y * z.y - z.z * z.z - z.w * z.w) + c.x;
		t.y = 2 * (z.x * z.y) + c.y;
		//t.z = 2 * (z.x * z.z) + c.z;
		//t.w = 2 * (z.x * z.w) + c.w;
		z = t;
	}*/

	//Bin search
	/*fFloat angle = atan2((fFloat)y, (fFloat)x);
	fFloat modulo = sqrt((z.x * z.x) + (z.y * z.y));
	int step = 1;
	for (int i = 0; i < 50000; ++i)
	{
		iterations += step;
		fFloat cosinus = cos(angle * (1 << iterations));
		cosinus *= cosinus;
		fFloat sinus = sin(angle * (1 << iterations));
		sinus *= sinus;
		fFloat currentValue = pow((fFloat)modulo, (fFloat)(1 << iterations))*sqrt(sinus + sinus);
		if (currentValue >= edge)
		{
			if (step <= 1) break;
			else
			{
				iterations -= step;
				step /= 2;
			}
		}
		else
		{
			if (step <= max_iterations)
				step *= 2;
			else break;
		}
	}*/
	//iterative
	int bailoutIndex = -1;
	//COMPLEX MADELVROT
	for (iterations = 0; iterations < max_iterations; ++iterations)
	{
		tmpz.x = z.x * z.x - z.y * z.y + c.x;
		tmpz.y = 2 * z.x * z.y + c.y;
		z.x = tmpz.x;
		z.y = tmpz.y;
		if (bailoutIndex != -1)
		if ((z.x * z.x + z.y * z.y) < edge)
			bailoutIndex = iterations;
	}
	if (bailoutIndex != -1)
	{
		//iterations -= bailoutIndex;
	}

	// (x+y)^5 = ax^2*y^2;

	/*float sum = powf(c.x + c.y, 5.0f);
	float mul = z.x * c.x*c.x * c.y*c.y;

	float difference = sum - mul;

	if (!difference)
		iterations = max_iterations;
	else
	{
		float alpha = powf(atanf(1.0f / difference) / 3.14159 * 2, colorExp);
		iterations = max_iterations * alpha;
	}*/


	// same parametrized
	/*float xval = z.x * c.x * c.x * powf(1 + c.x, 5);
	float yval = z.x * c.y * c.y * powf(1 + c.y, 5);

	float tolerance = fabsf(c.x - xval) * fabsf(c.y - yval);
	if (!tolerance)
		iterations = max_iterations;
	else
	{
		float alpha = powf(atanf(1.0f / tolerance) / 3.14159 * 2, colorExp);
		iterations = max_iterations * alpha;
	}*/


	uchar4 * currentPixel = pixels + (x + y * blockDim.x);

	float3 resultColor;
	if (iterations < max_iterations)
	{
		float coeff = (float)(max_iterations - iterations) / max_iterations;

		resultColor =
		{
			powf(colorLow.x * (1 - coeff) + colorHigh.x * coeff, colorExp),
			powf(colorLow.y * (1 - coeff) + colorHigh.y * coeff, colorExp),
			powf(colorLow.z * (1 - coeff) + colorHigh.z * coeff, colorExp)
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

__global__ void VolMandelbrotFractalDraw(fFloat xmin, fFloat xmax, fFloat ymin, fFloat ymax, VolFractalDataStruct d, uchar4 * pixels)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	unsigned int iterations = 0;

	//Planar projection

	double3 fwdVec = d.dir;
	double3 rightVec = crossVec(fwdVec, {0.,0.,1.});
	double3 upVec = crossVec(rightVec, fwdVec);
	normalizeVec(&rightVec);
	normalizeVec(&upVec);
	normalizeVec(&fwdVec);

	scaleVec(&fwdVec, 1.0 / tan(d.FOV/2));

	double alphaX = (double)x / blockDim.x;
	alphaX = xmin * (1 - alphaX) + xmax * alphaX;
	alphaX = (alphaX - 0.5) * 2;
	double alphaY = (double)y / gridDim.x;
	alphaY = ymin * (1 - alphaY) + ymax * alphaY;
	alphaY = (alphaY - 0.5) * 2;

	double3 stepVec = fwdVec;
	scaleVec(&rightVec, alphaX);
	scaleVec(&upVec, alphaY);
	addVec(&stepVec, rightVec);
	addVec(&stepVec, upVec);
	normalizeVec(&stepVec);
	//scaleVec(&stepVec, d.minStep);

	//End planar projection
	double3 p = d.loc;

	//Raymarcher
	double alpha = Raymarcher(stepVec, p, d.max_iterations, d.additionalNum1, d.additionalNum2).w;

	/*double4 rayResult = Raymarcher(stepVec, p, d.max_iterations, d.additionalNum1, d.additionalNum2);
	double3 normal = getDENorm({ rayResult.x, rayResult.y, rayResult.z }, d.additionalNum1, d.additionalNum2, &HashmapDE);
	double3 lightDir = {0.25, 1.0, -1.0};
	normalizeVec(&lightDir);
	double alpha = 0;
	if (Raymarcher(
	{ -lightDir.x, -lightDir.y, -lightDir.z }, 
	{ rayResult.x - lightDir.x * NormalDelta, rayResult.y - lightDir.y * NormalDelta, rayResult.z - lightDir.z * NormalDelta },
	d.max_iterations, d.additionalNum1, d.additionalNum2).w
	>= 0)
	alpha = -dotVec(lightDir, normal);*/
	/*for (int i = 0; i < d.raySteps; ++i)
	{
		scaleVec(&stepVec, d.stepMul);
		addVec(&p, stepVec);


		//Cubescan
		/*if (p.x >= -1 && p.x <= 1 &&
			p.y >= -1 && p.y <= 1 &&
			p.z >= -1 && p.z <= 1)
			alpha *= 0.99f;*/

		//Spherescan
		/*if (p.x * p.x + p.y * p.y + p.z * p.z < 1)
			alpha += 0.01f;*/

		//MANDERBROT
		/*double2 z = { p.z, d.additionalNum1 };
		double2 c = { p.x, p.y };
		double2 t;

		for (iterations = 0; iterations < d.max_iterations && (z.x * z.x + z.y * z.y) < d.edge; ++iterations)
		{
			t.x = (z.x * z.x - z.y * z.y) + c.x;
			t.y = 2 * (z.x * z.y) + c.y;
			z = t;
		}

		if (iterations < d.max_iterations)
		{
			float coeff = (float)(d.max_iterations - iterations) / d.max_iterations;
			alpha += coeff * d.brightness;
		}
		else if (d.stopOnHit)
			break;*/
	//}
	/*if (rayResult.w > 0)
	{
		if (alpha > 0)
		*(pixels + (x + y * blockDim.x)) =
		{
			powf(d.colorLow.x * (1-alpha)+d.colorHigh.x * (alpha), d.colorExp) * 255,
			powf(d.colorLow.y * (1-alpha)+d.colorHigh.y * (alpha), d.colorExp) * 255,
			powf(d.colorLow.z * (1-alpha)+d.colorHigh.z * (alpha), d.colorExp) * 255,
			255
		};
		else
		*(pixels + (x + y * blockDim.x)) =
		{
			powf(d.colorLow.x, d.colorExp) * 255,
			powf(d.colorLow.y, d.colorExp) * 255,
			powf(d.colorLow.z, d.colorExp) * 255,
			255
		};
	}*/

	/*if (alpha > 0)
		*(pixels + (x + y * blockDim.x)) =
	{
		powf(d.colorLow.x * (1 - alpha) + d.colorHigh.x * (alpha), d.colorExp) * 255,
		powf(d.colorLow.y * (1 - alpha) + d.colorHigh.y * (alpha), d.colorExp) * 255,
		powf(d.colorLow.z * (1 - alpha) + d.colorHigh.z * (alpha), d.colorExp) * 255,
		255
	};
	else if (false)
		*(pixels + (x + y * blockDim.x)) = 
	{
		powf(d.colorLow.x * (1 + alpha) + d.colorHigh.x * (-alpha), d.colorExp) * 255,
		powf(d.colorLow.y * (1 + alpha) + d.colorHigh.y * (-alpha), d.colorExp) * 255,
		powf(d.colorLow.z * (1 + alpha) + d.colorHigh.z * (-alpha), d.colorExp) * 255,
		255
	};
	else
		*(pixels + (x + y * blockDim.x)) = 
	{
		35,
		25,
		20,
		255
	};*/
}

__global__ void ClearBuffer(char4 * pbo)
{
	pbo[threadIdx.x + blockIdx.x * blockDim.x] = { 127, 0, 127, 255 };
}

__global__ void SupersampleOnce(int factor, uchar4 * from, uchar4 * to)
{
	int basePixel = threadIdx.x + blockIdx.x * (blockDim.x * factor) * 2;
	
	float4 color = { 0, 0, 0, 0 };

	int targetPixel = threadIdx.x * 2 + blockIdx.x * (blockDim.x * factor) * 4;
	color.x += from[targetPixel].x; 
	color.y += from[targetPixel].y; 
	color.z += from[targetPixel].z; 
	color.w += from[targetPixel].w;

	targetPixel += 1;
	color.x += from[targetPixel].x; 
	color.y += from[targetPixel].y; 
	color.z += from[targetPixel].z;
	color.w += from[targetPixel].w;

	targetPixel += (blockDim.x * factor) * 2;
	color.x += from[targetPixel].x; 
	color.y += from[targetPixel].y; 
	color.z += from[targetPixel].z; 
	color.w += from[targetPixel].w;

	targetPixel -= 1;
	color.x += from[targetPixel].x; 
	color.y += from[targetPixel].y; 
	color.z += from[targetPixel].z; 
	color.w += from[targetPixel].w;


	color.x /= 4.0f; color.y /= 4.0f; color.z /= 4.0f; color.w /= 4.0f;

	to[basePixel] = uchar4 { color.x, color.y, color.z, color.w };
}

__global__ void SubsampleOnce(char4 * from, char4 * to)
{
	int targetPixel = threadIdx.x + blockIdx.x * blockDim.x;
	int basePixel = threadIdx.x / 2 + (blockIdx.x / 2) * (blockDim.x / 2);

	to[targetPixel] = from[basePixel];
}

__global__ void Inject(int mainFactor, int sideFactor, int x, int y, char4 * from, char4 * to)
{
	int basePixel = threadIdx.x + blockIdx.x * blockDim.x * mainFactor;
	int targetPixel = threadIdx.x + x + (blockIdx.x + y) * blockDim.x * sideFactor;
	to[targetPixel] = from[basePixel];
}

__global__ void Render(int stretchFactor, char4 * from, char4 * to)
{
	int targetPixel = threadIdx.x + blockIdx.x * blockDim.x;
	int basePixel = threadIdx.x / stretchFactor + (blockIdx.x / stretchFactor) * (blockDim.x);

	to[targetPixel] = from[basePixel];
}

void DrawProgressiveFractal(FractalDataStruct * fractalData)
{
	if (!progressiveFractalData.frameBuffer)
	{
		cudaMalloc(&progressiveFractalData.frameBuffer, 4 * 1024 * 1024);
		ClearBuffer<<<1024, 1024>>>((char4*)progressiveFractalData.frameBuffer);
	}
	if (!progressiveFractalData.blockBuffer)
	{
		cudaMalloc(&progressiveFractalData.blockBuffer, 4 * blockSize * blockSize);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveFractalData.blockBuffer);
	}
	if (!progressiveFractalData.injectionBuffer)
	{
		cudaMalloc(&progressiveFractalData.injectionBuffer, 4 * blockSize * blockSize);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveFractalData.injectionBuffer);
	}
	

	fdata.vboPointer = fractalData->vboPointer;

	if (isDataChanged(fractalData))
	{
		fdata = *fractalData;
		ClearBuffer<<<1024, 1024>>>((char4*)progressiveFractalData.frameBuffer);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveFractalData.blockBuffer);

		pfdata.currentRenderingTesselation = 1;
		pfdata.renderingBlockNum = 0;

		MandelbrotFractalDraw<<<blockSize, blockSize>>>(
			fdata.xmin, fdata.xmax,
			fdata.ymin, fdata.ymax,
			fdata.max_iterations,
			fdata.edge,
			fdata.additionalNum1, fdata.additionalNum2,
			fdata.colorLow, fdata.colorHigh, fdata.colorExp,
			(uchar4*)pfdata.blockBuffer);

		Inject<<<blockSize, blockSize>>>(1, 1024 / blockSize, 0, 0, (char4*)pfdata.blockBuffer, (char4*)pfdata.frameBuffer);
		Render<<<1024, 1024>>>(1024 / blockSize, (char4*)pfdata.frameBuffer, (char4*)fdata.vboPointer);
	}
	else if (pfdata.currentRenderingTesselation < MaxRenderingTesselation)
	{
		int tessPow = 1 << (pfdata.currentRenderingTesselation - 1);

		int blockX = pfdata.renderingBlockNum % tessPow;
		int blockY = pfdata.renderingBlockNum / tessPow;

		const int sizeRatio = 1024 / blockSize;

		int scaleFactor = sizeRatio / tessPow;
		if (scaleFactor < 1) scaleFactor = 1;

		int injectionRatio = 1;
		if (tessPow > sizeRatio) injectionRatio = tessPow / sizeRatio;

		fFloat xDif = (fdata.xmax - fdata.xmin) / tessPow;
		fFloat yDif = (fdata.ymax - fdata.ymin) / tessPow;

		MandelbrotFractalDraw<<<blockSize, blockSize>>>(
			fdata.xmin + xDif * blockX, fdata.xmin + xDif * (blockX + 1),
			fdata.ymin + yDif * blockY, fdata.ymin + yDif * (blockY + 1),
			fdata.max_iterations,
			fdata.edge,
			fdata.additionalNum1, fdata.additionalNum2,
			fdata.colorLow, fdata.colorHigh, fdata.colorExp,
			(uchar4*)pfdata.blockBuffer);
		for (int i = 1; i < injectionRatio; i *= 2)
		{
			if (blockX == 0 && blockY == 0)
				std::cout << i << "\t" << blockSize / i << "\t" << i << std::endl;
			SupersampleOnce<<<blockSize / (i<<1), blockSize / (i<<1)>>>(i, (uchar4*)pfdata.blockBuffer, (uchar4*)pfdata.injectionBuffer);
			cudaMemcpy((char4*)pfdata.blockBuffer, (char4*)pfdata.injectionBuffer, 4 * blockSize * blockSize, cudaMemcpyDeviceToDevice);
		}

		Inject<<<blockSize / injectionRatio, blockSize / injectionRatio>>>
			(injectionRatio,
			sizeRatio * injectionRatio,
			blockX * blockSize / injectionRatio, 
			blockY * blockSize / injectionRatio, 
			(char4*)pfdata.blockBuffer, 
			(char4*)pfdata.frameBuffer);

		Render<<<1024, 1024>>>(scaleFactor, (char4*)pfdata.frameBuffer, (char4*)fdata.vboPointer);
	}

	if (pfdata.renderingBlockNum >=
		(1 << (pfdata.currentRenderingTesselation - 1)) * (1 << (pfdata.currentRenderingTesselation - 1)) - 1)
	{
		pfdata.renderingBlockNum = 0;
		++pfdata.currentRenderingTesselation;
		if (pfdata.currentRenderingTesselation > MaxRenderingTesselation)
			pfdata.currentRenderingTesselation = MaxRenderingTesselation;
		else
		{
			int scaleFactor = 1024 / blockSize / (1 << (pfdata.currentRenderingTesselation - 1));
			if (scaleFactor >= 1)
			{
				Render<<<1024, 1024>>>(2, (char4*)pfdata.frameBuffer, (char4*)fdata.vboPointer);
				cudaMemcpy(pfdata.frameBuffer, fdata.vboPointer, 4 * 1024 * 1024, cudaMemcpyDeviceToDevice);
				Render<<<1024, 1024>>>(scaleFactor, (char4*)pfdata.frameBuffer, (char4*)fdata.vboPointer);
			}
		}
	}
	else
	{
		++pfdata.renderingBlockNum;
	}
}
void DrawProgressiveVolFractal(VolFractalDataStruct * fractalData)
{
	if (!progressiveVolFractalData.frameBuffer)
	{
		cudaMalloc(&progressiveVolFractalData.frameBuffer, 4 * 1024 * 1024);
		ClearBuffer<<<1024, 1024>>>((char4*)progressiveVolFractalData.frameBuffer);
	}
	if (!progressiveVolFractalData.blockBuffer)
	{
		cudaMalloc(&progressiveVolFractalData.blockBuffer, 4 * blockSize * blockSize);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveVolFractalData.blockBuffer);
	}
	if (!progressiveVolFractalData.injectionBuffer)
	{
		cudaMalloc(&progressiveVolFractalData.injectionBuffer, 4 * blockSize * blockSize);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveVolFractalData.injectionBuffer);
	}


	vfdata.pixels = fractalData->pixels;

	if (isDataChanged(fractalData))
	{
		vfdata = *fractalData;
		ClearBuffer<<<1024, 1024>>>((char4*)progressiveVolFractalData.frameBuffer);
		ClearBuffer<<<blockSize, blockSize>>>((char4*)progressiveVolFractalData.blockBuffer);

		vpfdata.currentRenderingTesselation = 1;
		vpfdata.renderingBlockNum = 0;

		VolMandelbrotFractalDraw<<<blockSize, blockSize>>>
			(0., 
			1., 
			0., 
			1., 
			vfdata, 
			(uchar4*)vpfdata.blockBuffer);

		Inject<<<blockSize, blockSize>>>(1, 1024 / blockSize, 0, 0, (char4*)vpfdata.blockBuffer, (char4*)vpfdata.frameBuffer);
		Render<<<1024, 1024>>>(1024 / blockSize, (char4*)vpfdata.frameBuffer, (char4*)vfdata.pixels);
	}
	else if (vpfdata.currentRenderingTesselation < MaxRenderingTesselation)
	{
		int tessPow = 1 << (vpfdata.currentRenderingTesselation - 1);

		int blockX = vpfdata.renderingBlockNum % tessPow;
		int blockY = vpfdata.renderingBlockNum / tessPow;

		const int sizeRatio = 1024 / blockSize;

		int scaleFactor = sizeRatio / tessPow;
		if (scaleFactor < 1) scaleFactor = 1;

		int injectionRatio = 1;
		if (tessPow > sizeRatio) injectionRatio = tessPow / sizeRatio;

		VolMandelbrotFractalDraw<<<blockSize, blockSize>>>(
			1. / tessPow * blockX, 1. / tessPow * (blockX + 1),
			1. / tessPow * blockY, 1. / tessPow * (blockY + 1),
			vfdata,
			(uchar4*)vpfdata.blockBuffer);

		for (int i = 1; i < injectionRatio; i *= 2)
		{
			if (blockX == 0 && blockY == 0)
				std::cout << i << "\t" << blockSize / i << "\t" << i << std::endl;
			SupersampleOnce<<<blockSize / (i << 1), blockSize / (i << 1)>>>(i, (uchar4*)vpfdata.blockBuffer, (uchar4*)vpfdata.injectionBuffer);
			cudaMemcpy((char4*)vpfdata.blockBuffer, (char4*)vpfdata.injectionBuffer, 4 * blockSize * blockSize, cudaMemcpyDeviceToDevice);
		}

		Inject<<<blockSize / injectionRatio, blockSize / injectionRatio>>>
			(injectionRatio,
			sizeRatio * injectionRatio,
			blockX * blockSize / injectionRatio,
			blockY * blockSize / injectionRatio,
			(char4*)vpfdata.blockBuffer,
			(char4*)vpfdata.frameBuffer);

		Render<<<1024, 1024>>>(scaleFactor, (char4*)vpfdata.frameBuffer, (char4*)vfdata.pixels);
	}

	if (vpfdata.renderingBlockNum >=
		(1 << (vpfdata.currentRenderingTesselation - 1)) * (1 << (vpfdata.currentRenderingTesselation - 1)) - 1)
	{
		vpfdata.renderingBlockNum = 0;
		++vpfdata.currentRenderingTesselation;
		if (vpfdata.currentRenderingTesselation > MaxRenderingTesselation)
			vpfdata.currentRenderingTesselation = MaxRenderingTesselation;
		else
		{
			int scaleFactor = 1024 / blockSize / (1 << (vpfdata.currentRenderingTesselation - 1));
			if (scaleFactor >= 1)
			{
				Render<<<1024, 1024>>>(2, (char4*)vpfdata.frameBuffer, (char4*)vfdata.pixels);
				cudaMemcpy(vpfdata.frameBuffer, vfdata.pixels, 4 * 1024 * 1024, cudaMemcpyDeviceToDevice);
				Render<<<1024, 1024>>>(scaleFactor, (char4*)vpfdata.frameBuffer, (char4*)vfdata.pixels);
			}
		}
	}
	else
	{
		++vpfdata.renderingBlockNum;
	}
}