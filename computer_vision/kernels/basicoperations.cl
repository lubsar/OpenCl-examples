__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void grayscale(__read_only image2d_t input,
                        __write_only image2d_t output) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 sample = read_imagef(input, sampler, pos);

  float gray = 0.299 * sample.x + 0.587 * sample.y + 0.114 * sample.z;

  write_imagef(output, pos, (float4)(gray, gray, gray, 1.0f));
}

__kernel void colorshift(__read_only image2d_t input,
                        __write_only image2d_t output,
                        const float4 value) {

  const int2 pos = {get_global_id(0), get_global_id(1)};

  write_imagef(output, pos, read_imagef(input, sampler, pos) + value);
}

__kernel void colorscale(__read_only image2d_t input,
                        __write_only image2d_t output,
                        const float4 value) {

  const int2 pos = {get_global_id(0), get_global_id(1)};

  write_imagef(output, pos, read_imagef(input, sampler, pos) * value);
}

__kernel void toHSV(__read_only image2d_t input,
                        __write_only image2d_t output) {

  const int2 pos = {get_global_id(0), get_global_id(1)};
  float4 pixel = read_imagef(input, sampler, pos);

  float h = 0.0f;
  float s = 0.0f;
  float v = 0.0f;

  //value is max(R,G,B)
  v = max(pixel.x, max(pixel.y, pixel.z));

  //C is value - min(R,G,B)
  float c = v - min(pixel.x, min(pixel.y, pixel.z));

  //saturation is C / value
  if(v != 0.0f) {
    s = c / v;
  } else {
    s = 0.0f;
  }

  if(c == 0.0f) {
    h = 0.0f;
  } else {
    //value is red

    if(pixel.x == v) {
      h = (pixel.y - pixel.z) / c;
    //value is blue
    } else if(pixel.y == v) {
      h = (pixel.z - pixel.x) / c + 2.0f;
    //value is green
    } else {
      h = (pixel.x - pixel.y) / c + 4.0f;
    }

    if(h < 0.0f) {
      h = h / 6.0f + 1.0f;
    } else {
      h = h / 6.0f;
    }
  }

  write_imagef(output, pos, (float4)(h,s,v, 1.0f));
}

__kernel void toRGB(__read_only image2d_t input,
                        __write_only image2d_t output) {

  const int2 pos = {get_global_id(0), get_global_id(1)};
  float4 hsv = read_imagef(input, sampler, pos);

  float4 RGB = (float4)(0.0f, 0.0f, 0.0f, 1.0f);

  int hi = (hsv.x * 6.0f);
  float f = hsv.x * 6.0f - hi;

  float p = hsv.z * (1.0f - hsv.y);
  float q = hsv.z * (1.0f - f * hsv.y);
  float t = hsv.z * (1.0f - (1.0f - f) * hsv.y);

  switch(hi) {
    case 0: { RGB = (float4)(hsv.z, t, p, 1.0f); break;}
    case 1: { RGB = (float4)(q, hsv.z, p, 1.0f); break;}
    case 2: { RGB = (float4)(p, hsv.z, t, 1.0f); break;}
    case 3: { RGB = (float4)(p, q, hsv.z, 1.0f); break;}
    case 4: { RGB = (float4)(t, p, hsv.z, 1.0f); break;}
    case 5: { RGB = (float4)(hsv.z, p, q, 1.0f); break;}
  }

  write_imagef(output, pos, RGB);
}
