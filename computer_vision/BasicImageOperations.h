#pragma once

#include "CLEngine.h"
#include "Image.h"

namespace CV {
	class BasicImageOperations : public CLEngine {
	public:
		BasicImageOperations();

		Image* grayscale(Image& input);
		Image* scaleColor(Image& input, cl_float4 factors);
		Image* shift(Image& input, cl_float4 value);
		Image* toHSV(Image& input);
		Image* toRGB(Image& input);
		Image* resizeNN(Image& input, int newWidth, int newHeight);
		Image* resizeBilinear(Image& input, int newWidth, int newHeight);

		Image* convolution(Image& input, float* convoKernel, int convoWidth, int convoHeight);
	};
}