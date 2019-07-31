#include "BasicImageOperations.h"

namespace CV {
	BasicImageOperations::BasicImageOperations() : CLEngine("kernels/basicoperations.cl") {
	}

	Image* BasicImageOperations::grayscale(Image& input) {
		cl::Kernel kernel = prepareKernel("grayscale");

		Image *output = new Image(input.width, input.height, input.channels);

		cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
		cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

		err = kernel.setArg(0, inBuff);
		err = kernel.setArg(1, outBuff);

		cl::size_t<3> origin;
		cl::size_t<3> region;
		region[0] = input.width;
		region[1] = input.height;
		region[2] = 1;

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
		err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

		return output;
	}

	Image* BasicImageOperations::scaleColor(Image& input, cl_float4 factors) {
		{
			cl::Kernel kernel = prepareKernel("colorscale");

			Image *output = new Image(input.width, input.height, input.channels);

			cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
			cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
			cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

			err = kernel.setArg(0, inBuff);
			err = kernel.setArg(1, outBuff);
			err = kernel.setArg(2, sizeof(cl_float4), &factors);

			cl::size_t<3> origin;
			cl::size_t<3> region;
			region[0] = input.width;
			region[1] = input.height;
			region[2] = 1;

			err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
			err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

			return output;
		}
	}

	Image* BasicImageOperations::shift(Image& input, cl_float4 value) {
		cl::Kernel kernel = prepareKernel("colorshift");

		Image *output = new Image(input.width, input.height, input.channels);

		cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
		cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

		err = kernel.setArg(0, inBuff);
		err = kernel.setArg(1, outBuff);
		err = kernel.setArg(2, sizeof(cl_float4), &value);

		cl::size_t<3> origin;
		cl::size_t<3> region;
		region[0] = input.width;
		region[1] = input.height;
		region[2] = 1;

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
		err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

		return output;
	}
	
	Image* BasicImageOperations::toHSV(Image& input) {
		cl::Kernel kernel = prepareKernel("toHSV");

		Image *output = new Image(input.width, input.height, input.channels);

		cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
		cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

		err = kernel.setArg(0, inBuff);
		err = kernel.setArg(1, outBuff);

		cl::size_t<3> origin;
		cl::size_t<3> region;
		region[0] = input.width;
		region[1] = input.height;
		region[2] = 1;

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
		err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

		return output;
	}
	
	Image* BasicImageOperations::toRGB(Image& input) {
		cl::Kernel kernel = prepareKernel("toRGB");

		Image *output = new Image(input.width, input.height, input.channels);

		cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
		cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

		err = kernel.setArg(0, inBuff);
		err = kernel.setArg(1, outBuff);

		cl::size_t<3> origin;
		cl::size_t<3> region;
		region[0] = input.width;
		region[1] = input.height;
		region[2] = 1;

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
		err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

		return output;
	}

	Image* BasicImageOperations::resizeNN(Image& input, int newWidth, int newHeight) {

	}

	Image* BasicImageOperations::convolution(Image& input, float* convoKernel, int convoWidth, int convoHeight) {
		cl::Kernel kernel = prepareKernel("convolution");

		Image *output = new Image(input.width, input.height, input.channels);

		cl::ImageFormat imgFormat(CL_RGBA, CL_UNORM_INT8);
		cl::Image2D inBuff(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, imgFormat, input.width, input.height, 0, input.buffer->data(), &err);
		cl::Image2D outBuff(this->context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, imgFormat, output->width, output->height, 0, output->buffer->data(), &err);

		err = kernel.setArg(0, inBuff);
		err = kernel.setArg(1, outBuff);

		cl::size_t<3> origin;
		cl::size_t<3> region;
		region[0] = input.width;
		region[1] = input.height;
		region[2] = 1;

		err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(input.width, input.height));
		err = queue.enqueueReadImage(outBuff, CL_TRUE, origin, region, 0, 0, output->buffer->data());

		return output;
	}
}