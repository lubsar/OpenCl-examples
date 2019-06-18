#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "CL/cl.hpp"
#include "lodepng/lodepng.h"

int main()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	/*for (auto platform : platforms) {
		std::vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		for (auto device : devices) {
			std::cout << device.getInfo<CL_DEVICE_VENDOR>() << " " << device.getInfo <CL_DEVICE_NAME>() << std::endl;
		}
	}*/

	cl::Platform platform = platforms.front();

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	cl::Device device = devices.front();

	std::vector<unsigned char> pixelData;
	unsigned int imageWidth, imageHeight;
	lodepng::decode(pixelData, imageWidth, imageHeight, "sample.png");

	//std::cout << (int)pixelData[0] << " " << (int)pixelData[1] << " "<< (int)pixelData[2] << std::endl;

	cl::Context context(device);
	cl::CommandQueue queue(context, device);
	cl_int err = CL_SUCCESS;

	std::ifstream kernelsrc("blur.cl");
	std::string srcstring(std::istreambuf_iterator<char>(kernelsrc), (std::istreambuf_iterator<char>()));
	//std::cout << srcstring << std::endl;
	cl::Program::Sources source(1, std::make_pair(srcstring.c_str(), srcstring.length() + 1));
	cl::Program program = cl::Program(context, source);

	try {
		program.build(devices);
	}
	catch (cl::Error error) {
		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		return 0;
	}

	float blurWeights[9] = {
		0.0f, 0.125f, 0.0f,
		0.125f, 0.5f, 0.125f,
		0.0f, 0.125f, 0.0f
	};

	unsigned char* output = new unsigned char[pixelData.size()];

	cl::Kernel blur(program, "blur", &err);

	cl::Buffer weights(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 9, &blurWeights, &err);
	cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
	cl::Image2D in(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, imageWidth, imageHeight, 0, pixelData.data(), &err);
	cl::Image2D out(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, imageWidth, imageHeight,0, output, &err);

	err = blur.setArg(0, in);
	err = blur.setArg(1, weights);
	err = blur.setArg(2, out);

	cl::size_t<3> origin;
	cl::size_t<3> region;
	region[0] = imageWidth;
	region[1] = imageHeight;
	region[2] = 1;

	err = queue.enqueueNDRangeKernel(blur, cl::NullRange, cl::NDRange(imageWidth, imageHeight));
	err = queue.enqueueReadImage(out, CL_TRUE, origin, region, 0, 0, output);

	lodepng::encode("result.png", output, imageWidth, imageHeight);

	std::cout << err << std::endl;

	return 0;
}
