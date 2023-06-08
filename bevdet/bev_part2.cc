/*
 * @Author: ycdhq 
 * @Date: 2023-06-08 10:25:02 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 11:11:14
 */

#include "bev_part2.h"

#include <algorithm>
#include <map>

#include "common/file.h"
#include "common/npy.h"
#include "common/math.h"

namespace bev {

using apollo::cyber::common::GetProtoFromFile;
using apollo::cyber::common::GetAbsolutePath;

bool BEVPart2::Init(const std::string& config_path) {
  BEVDet configs;

  AINFO << config_path;
  if (!GetProtoFromFile(config_path, &configs)) {
    return false;
  }

  auto part2_config = configs.part2_config();

  engine_file_ = part2_config.engine_file(); 
  // compute image provider parameters
  src_height_ = static_cast<uint16_t>(part2_config.src_height());
  src_width_ = static_cast<uint16_t>(part2_config.src_width());


  device_id_ = part2_config.gpu_id();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id_);
  AINFO << "GPU: " << prop.name;

  const auto net_param = part2_config.net_config();

  for (int i = 0; i < net_param.input_names_size(); ++i) {
    AINFO << net_param.input_names(i);
    input_names_.push_back(net_param.input_names(i)); 
  }
  for (int i = 0; i < net_param.output_names_size(); ++i) {
    AINFO << net_param.output_names(i);
    output_names_.push_back(net_param.output_names(i)); 
  }
  RTEngine::Init();
  return true;
}

bool BEVPart2::Detect(float* bev_feature) {
  auto start = std::chrono::high_resolution_clock::now();

  auto input_blob = RTEngine::get_blob(input_names_[0]);

  memcpy(input_blob->mutable_cpu_data(), bev_feature, 80*128*128 * sizeof(float));

  ADEBUG << "resize gpu finish.";
  cudaDeviceSynchronize();
  RTEngine::Infer();
  cudaDeviceSynchronize();
  AINFO << "part2 infer finish.";

  for (int i = 0; i < output_names_.size(); i++) {
    auto output_blob = RTEngine::get_blob(output_names_[i]);
    int out_b = output_blob->num();
    int out_c = output_blob->channels();
    int out_h = output_blob->height();
    int out_w = output_blob->width();
    float* output_data = output_blob->mutable_cpu_data();
    AINFO << out_b << " " << out_c << " " << out_h << " " << out_w;

#if 1
    std::string np_name = "../data/" + output_names_[i] + ".npy";
    AINFO << np_name;
    std::vector<unsigned long> npy_shape;
    std::vector<float> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    npy::LoadArrayFromNumpy(np_name, npy_shape, is_fortran, npy_vector);
    for (int z = 0; z < npy_vector.size(); z++) {
      AINFO << output_data[z] << " " << npy_vector[z];
    }

  }
#endif 
  return true;
}
}  // namespace bev
