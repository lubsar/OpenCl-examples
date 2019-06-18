__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void blur(__read_only image2d_t input,
                   __constant float* weights,
                   __write_only image2d_t output) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 sum = (float4)(0.0f);
  for(int i = -1; i <= 1; i++) {
    for(int o = -1; o <= 1; o++) {
      sum += weights[(o+1) + (i +1)*3] * read_imagef(input, sampler, pos + (int2)(o,i));
    }
  }

  write_imagef(output, pos, sum);
}
