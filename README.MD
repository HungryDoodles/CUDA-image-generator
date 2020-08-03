That's one ugly looking code and it barely works. Yet it was the second program I made using CUDA, don't judge.

Uses freeglut, glew, glfw and CUDA.
Progressively draws the image onto 1024 by 1024 canvas (because I was too lazy to properly go over dimension limitation by dividing grid).
Draws small boxes (128 by 128, if I remember correctly) and projects them onto canvas that allows to:
1) quickly navigate in the space of the object with moderate performance;
2) go beyond 1024x1024 given enough time by supersampling while stationary.
More like TXAA, but at larger scales and unnecessarily high precision.

For demonstrative purposes the executable has already been built and can be located in Release folder (_2.exe it is).
X - cycle between modes (1st mode is proof of work and can should be glitchy waveform, 2nd and 3rd are broken, 4th is Mandelbulb).
WASD - disgusting way of BIOS input to control camera.
Hold LMB - control camera.
Hold RMD - control 2-axis parameter of current mode.
'<' and '>' - decrease and increase number of iterations (affects all modes).
N and M - tonemapping; becomes quite peculiar going below 0.

IMPORTANT: When the 4th mode is activated the image should become brown-ish there will be one tiny pixel in the middle. To start seeing the image you need to press '>' a few times to increase number of iterations to something realistic and move the mouse to the right while holding RMB, which will increase circular parameter of the fractal.
Then you're free to explore.
Vertical movement of the mouse while holding RMB is reliable to the "gravity" effect of the raymarcher. This is trippy and disgusting, as you cannot escape the fractal by going backwards MWU-HA-HA-HA ahem...
