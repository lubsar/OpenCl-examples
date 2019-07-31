#include <string>
#include <iostream>
#include <vector>

#include "Image.h"
#include "lodepng/lodepng.h"

namespace CV {
	Image::Image(std::string path) {
		this->buffer = new std::vector<unsigned char>();

		lodepng::decode(*this->buffer, this->width, this->height, path);

		this->channels = 4;
	}

	Image::Image(int width, int height, int channels) {
		this->channels = channels;
		this->width = width;
		this->height = height;

		this->buffer = new std::vector<unsigned char>(width * height * channels);
	}

	void Image::setPixel(int x, int y, int channel, float value) {
		if (x < 0 || y < 0 || x >= this->width || y >= this->height) {
			return;
		}
		(*buffer)[(x + y * width) * channels + channel] = (unsigned char)(value * 255.0f);
	}

	//with clamping
	float Image::getPixel(int x, int y, int channel) {
		if (x < 0) {
			x = 0;
		}
		if (y < 0) {
			y = 0;
		}
		if (x >= this->width) {
			x = this->width - 1;
		}
		if (y >= this->height) {
			y = this->height - 1;
		}

		return ((*buffer)[(x + y * width) * channels] / 255.0f);
	}

	void Image::save(std::string path) {
		if (channels == 3) {
			std::vector<unsigned char>* outputBuffer = new std::vector<unsigned char>(this->width * this->height * 4);
			
			for (int channel = 0; channel < channels; channel++) {
				for (unsigned int i = 0; i < (this->width * this->height); i++) {
					std::cout << i * 3 + channel << std::endl;
					(*outputBuffer)[i * 4 + channel] = (*buffer)[i * 3 + channel];
				}
			}

			//add alpha channel with 255
			for (int i = 0; i < (this->width * this->height); i++) {
				(*outputBuffer)[i * 4 + 3] = 255;
			}
			
			lodepng::encode(path, *outputBuffer, this->width, this->height);

			delete outputBuffer;
		} else {
			lodepng::encode(path, *buffer, this->width, this->height);
		}
	}

	Image::~Image() {
		delete buffer;
	}
}
