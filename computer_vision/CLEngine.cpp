#include <iostream>
#include <fstream>
#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include "CLEngine.h"

namespace CV {
	CLEngine::CLEngine(std::string sourceFile) {
		this->platform = cl::Platform::getDefault();
		this->devices = std::vector<cl::Device>();
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

		this->device = devices.front();

		this->context = cl::Context(device);
		this->queue = cl::CommandQueue(context, device);
		this->err = CL_SUCCESS;

		//std::cout << device.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>() << std::endl;
		//return 0;

		std::ifstream kernelsrc(sourceFile);
		std::string srcstring(std::istreambuf_iterator<char>(kernelsrc), (std::istreambuf_iterator<char>()));
		//std::cout << srcstring << std::endl;
		cl::Program::Sources source(1, std::make_pair(srcstring.c_str(), srcstring.length() + 1));
		
		this->program = cl::Program(context, source);

		try {
			program.build(devices);
		}
		catch (cl::Error error) {
			std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		}
	}

	CLEngine::~CLEngine() {

	}

	cl::Kernel CLEngine::prepareKernel(std::string kernelName) {
		return cl::Kernel(program, kernelName.c_str(), &err);
	}

	int CLEngine::getErr() {
		return this->err;
	}
}
