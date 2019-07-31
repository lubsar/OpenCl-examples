#define __CL_ENABLE_EXCEPTIONS

#include "BasicImageOperations.h"

int main()
{
	CV::Image input("dog.png");
	
	CV::BasicImageOperations bio;

	//bio.shift(input, {0.0f, 0.4f, 0.4f, 0.0f})->save("output/shifted.png");
	//bio.grayscale(input)->save("output/gray.png");
	CV::Image* a = bio.toHSV(input);
	a->save("output/hsv.png");
	a = bio.scaleColor(*a, { 1.0, 2.0f, 1.0f, 1.0f });
	bio.toRGB(*a)->save("output/saturated.png");
}
