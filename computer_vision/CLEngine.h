#pragma once
#include <CL/cl.hpp>
#include <string>

namespace CV {
	class CLEngine {
	protected:
		cl::Platform platform;
		std::vector<cl::Device> devices;
		cl::Device device;
		cl::Context context;
		cl::CommandQueue queue;
		cl_int err;
		cl::Program program;

	public:
		CLEngine(std::string sourceFile);
		~CLEngine();
		int getErr();
		cl::Kernel prepareKernel(std::string kernelName);
	};
}