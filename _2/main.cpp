#include "glew.h"
#include "freeglut.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <ctime>
#include "CUDAPictureCalculator.cuh"
#include "SimpleVector2d.h"

#define VBOsize 1024 * 1024 * 4

using namespace std;

Mode workMode;
bool isLMB, isRMB;

//superSine additive data
clock_t startTimer;
//superSine

//fractal additive data
FractalDataStruct frac;
vec2<fFloat> screenPosTarget;
fFloat screenSizeTarget;
vec2<fFloat> screenPos;
fFloat screenSize;
vec2<fFloat> mouseOldPos;
//fractal

//volFractal data
VolFractalDataStruct vfrac;
double3 location = { -2., 0., 0. };
double3 smoothedLocation = { 0., 0., 0. };
double2 rotation = { 0., 0. };
double2 smoothedRotation = { 0., 0. };
fFloat travelSpeed = 0.05f;
//volFractal

GLuint vbo;
cudaGraphicsResource * cudaVBO;

void display()
{
	void * devPointer;
	size_t size;

	cudaGraphicsMapResources(1, &cudaVBO);
	cudaGraphicsResourceGetMappedPointer(&devPointer, &size, cudaVBO);

	clock_t frameTimer = clock();
	SuperSineDataStruct superSineData;

	double3 forwardVec = headingVec(rotation.x, rotation.y);;

	switch (workMode)
	{
	case superSine:
		superSineData.time = (float)(frameTimer - startTimer) / CLOCKS_PER_SEC;
		superSineData.vboPointer = devPointer;
		PictureCalculator(superSine, &superSineData);
		break;
	case fractal:
		frac.vboPointer = devPointer;
		PictureCalculator(fractal, (void*)&frac);
		break;
	case progressiveFractal:
		frac.vboPointer = devPointer;
		PictureCalculator(progressiveFractal, (void*)&frac);
		break;
	case volFractal:
		vfrac.loc = smoothedLocation;
		vfrac.dir = forwardVec;
		vfrac.pixels = (uchar4*)devPointer;
		PictureCalculator(volFractal, (void*)&vfrac);
		break;
	}

	cudaGraphicsUnmapResources(1, &cudaVBO);

	glClearColor(0.2, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	//glRasterPos2i(0, 0);
	glDisable(GL_DEPTH_TEST);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glDrawPixels(1024, 1024, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glutSwapBuffers();
}

void idle()
{
	fFloat alpha = 0.6;
	fFloat halfScreenSize;
	switch (workMode)
	{
	case progressiveFractal:
	case fractal:
		screenPos = vecLerp<fFloat>(screenPos, screenPosTarget, alpha);
		screenSize = screenSize * (1 - alpha) + screenSizeTarget * alpha;

		halfScreenSize = screenSize / 2;

		frac.xmax = screenPos.x + halfScreenSize;
		frac.xmin = screenPos.x - halfScreenSize;
		frac.ymax = screenPos.y + halfScreenSize;
		frac.ymin = screenPos.y - halfScreenSize;
		break;
	case volFractal:
		smoothedLocation.x = smoothedLocation.x * (1 - alpha) + location.x * alpha;
		smoothedLocation.y = smoothedLocation.y * (1 - alpha) + location.y * alpha;
		smoothedLocation.z = smoothedLocation.z * (1 - alpha) + location.z * alpha;

		smoothedRotation.x = smoothedRotation.x * (1 - alpha) + rotation.x * alpha;
		smoothedRotation.y = smoothedRotation.y * (1 - alpha) + rotation.y * alpha;
		break;
	}
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	double3 forward = headingVec(rotation.x, rotation.y);
	double3 right = crossVec(forward, { 0., 0., 1. });
	double3 up = crossVec(right, forward);
	normalizeVec(&forward);
	normalizeVec(&right);
	normalizeVec(&up);
	scaleVec(&forward, travelSpeed);
	scaleVec(&right, travelSpeed);
	scaleVec(&up, travelSpeed);
	switch (key)
	{
	case 'x':
		cout << "Mode switch\n";
		workMode = (Mode)(((int)workMode + 1) % ModeEnumCount);
		break;
	case '.': //>
		cout << "Fractal iterations ++ " << frac.max_iterations;
		frac.max_iterations += 100;
		vfrac.max_iterations += 5;
		cout << "->" << frac.max_iterations << "/" << vfrac.max_iterations << "\n";
		break;
	case ',': //<
		cout << "Fractal iterations -- " << frac.max_iterations;
		frac.max_iterations -= 100;
		vfrac.max_iterations -= 5;
		if (frac.max_iterations <= 0) frac.max_iterations = 1;
		if (vfrac.max_iterations <= 0) vfrac.max_iterations = 1;
		cout << "->" << frac.max_iterations << "/" << vfrac.max_iterations << "\n";
		break;
	case ']':
		cout << "Fractal edge -- " << frac.edge;
		frac.edge += 0.05f;
		vfrac.edge = frac.edge;
		cout << "->" << frac.edge << "\n";
		break;
	case '[':
		cout << "Fractal edge ++ " << frac.edge;
		frac.edge -= 0.05f;
		vfrac.edge = frac.edge;
		cout << "->" << frac.edge << "\n";
		break;
	case 'm':
		cout << "Color curve ++ " << frac.colorExp;
		frac.colorExp += 0.15f;
		vfrac.colorExp = frac.colorExp;
		cout << "->" << frac.colorExp << "\n";
		break;
	case 'n':
		cout << "Color curve -- " << frac.colorExp;
		frac.colorExp -= 0.15f;
		vfrac.colorExp = frac.colorExp;
		cout << "->" << frac.colorExp << "\n";
		break;
	case 'w':
		location.x += forward.x;
		location.y += forward.y;
		location.z += forward.z;
		break;
	case 's':
		location.x -= forward.x;
		location.y -= forward.y;
		location.z -= forward.z;
		break;
	case 'a':
		location.x -= right.x;
		location.y -= right.y;
		location.z -= right.z;
		break;
	case 'd':
		location.x += right.x;
		location.y += right.y;
		location.z += right.z;
		break;
	case 'q':
		location.x += up.x;
		location.y += up.y;
		location.z += up.z;
		break;
	case 'z':
		location.x -= up.x;
		location.y -= up.y;
		location.z -= up.z;
		break;
	case 27:
		glutLeaveMainLoop();
		break;
	}
}

void mouse(int key, int state, int x, int y)
{
	switch (key)
	{
	case 0: //LMB
		if (state == GLUT_DOWN)
		{
			mouseOldPos = vec2<fFloat>(x, 1024 - y);
			isLMB = true;
		}
		else if (state == GLUT_UP)
			isLMB = false;
		break;
	case 2: //RMB
		if (state == GLUT_DOWN)
		{
			mouseOldPos = vec2<fFloat>(x, 1024 - y);
			isRMB = true;
		}
		else if (state == GLUT_UP)
			isRMB = false;
		break;
	case 3: //MW UP
		if (state == GLUT_DOWN) screenSizeTarget *= 0.75;
		cout << "Size " << screenSizeTarget << "\n";
		break;
	case 4: //MW DOWN
		if (state == GLUT_DOWN) screenSizeTarget *= 1 / 0.75;
		cout << "Size " << screenSizeTarget << "\n";
		break;
	}
}

void mouseMove(int x, int y)
{
	y = 1024 - y;
	if (isLMB)
	{
		auto mouseDelta = vec2<fFloat>(x, y) - mouseOldPos;
		screenPosTarget = screenPosTarget - mouseDelta * (screenSize / 1024);
		rotation.x -= mouseDelta.x / 280.;
		rotation.y += mouseDelta.y / 280.;
	}
	if (isRMB)
	{
		frac.additionalNum1 += (fFloat)(x - mouseOldPos.x) / 10000 * screenSize;
		frac.additionalNum2 += (fFloat)(y - mouseOldPos.y) / 10000 * screenSize;

		vfrac.additionalNum1 += (fFloat)(x - mouseOldPos.x) / 10000 * screenSize;
		vfrac.additionalNum2 += (fFloat)(y - mouseOldPos.y) / 10000 * screenSize;
		//cout << "AdditionNum1 " << frac.additionalNum1 << "\n";
		//cout << "AdditionNum2 " << frac.additionalNum2 << "\n";
	}
	mouseOldPos = vec2<fFloat>(x, y);
}

int main(int argc, char**argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(1024, 1024);
	glutCreateWindow("Window");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseMove);
	glutIdleFunc(idle);
	glewInit();

	cudaSetDevice(0);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, VBOsize, NULL, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cudaVBO, vbo, cudaGraphicsMapFlagsNone);

	startTimer = clock();
	workMode = superSine;
	frac.colorHigh = { 1.0f, 1.0f, 1.0f };
	frac.colorLow = {0.0f, 0.0f, 0.0f};
	frac.edge = 4.0f;
	frac.max_iterations = 15; 
	frac.additionalNum1 = 0;
	frac.additionalNum2 = 0;
	frac.xmax = 1.0f;
	frac.xmin = -1.0f;
	frac.ymax = 1.0f;
	frac.ymin = -1.0f;
	frac.colorExp = 1;
	isLMB = false;
	isRMB = false;

	vfrac.additionalNum1 = 1.f;
	vfrac.additionalNum2 = 0.f;
	vfrac.brightness = 0.01f;
	vfrac.colorExp = 1.f;
	vfrac.colorHigh = { 1.0f, 0.9f, 0.8f };
	vfrac.colorLow = { 0.1f, 0.08f, 0.05f };
	vfrac.dir = { -1, 0, 0 };
	vfrac.edge = 2.0f;
	vfrac.FOV = 1.5f;
	vfrac.loc = {-2, 0, 0 };
	vfrac.max_iterations = 5;
	vfrac.pixels = nullptr;
	vfrac.raySteps = 50;
	vfrac.stopOnHit = true;
	vfrac.minStep = 0.01;
	vfrac.stepMul = 1.000001;

	screenSizeTarget = 10;
	screenSize = 10;
	mouseOldPos = vec2<fFloat>(-1, -1);

	fractalInit();

	glutMainLoop();

	//cudaFree(progressiveFractalData.frameBuffer);
	//cudaFree(progressiveFractalData.blockBuffer);
	//cudaFree(progressiveFractalData.injectionBuffer);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, vbo);
	glDeleteBuffers(1, &vbo);

	return 0;
}