#pragma once

namespace CV {
	//image data is written per channel, row major, 0-1f
	struct Image {
		unsigned int width, height, channels;
		std::vector<unsigned char>* buffer;

		Image(std::string path);
		Image(int width, int height, int channels);
		~Image();

		void setPixel(int x, int y, int channel, float value);
		void toHSV();
		void toRGB();

		//with clamping
		float getPixel(int x, int y, int channel);
		void save(std::string path);
	};
}