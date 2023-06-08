/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 10:45:30 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 09:33:19
 */

#include "bev_part1.h"

#include <algorithm>
#include <map>

#include "common/file.h"
#include "common/npy.h"
#include "common/math.h"

namespace bev {

using apollo::cyber::common::GetProtoFromFile;
using apollo::cyber::common::GetAbsolutePath;

bool BEVPart1::Init(const std::string& config_path) {
  BEVDet configs;

  AINFO << config_path;
  if (!GetProtoFromFile(config_path, &configs)) {
    return false;
  }

  auto part1_config = configs.part1_config();

  engine_file_ = part1_config.engine_file(); 
  // compute image provider parameters
  src_height_ = static_cast<uint16_t>(part1_config.src_height());
  src_width_ = static_cast<uint16_t>(part1_config.src_width());
  resize_height_ = static_cast<uint16_t>(part1_config.resize_height());
  resize_width_ = static_cast<uint16_t>(part1_config.resize_width());  

  input_offset_y_ = static_cast<uint16_t>(part1_config.input_offset_y());
  input_offset_x_ = static_cast<uint16_t>(part1_config.input_offset_x());

  // input_offset_y_ = int(src_height_ * 236 / 590);
  // input_offset_x_ = 0;

  crop_height_ = src_height_ - input_offset_y_;
  crop_width_ = src_width_ - input_offset_x_;
  device_id_ = part1_config.gpu_id();
  if (part1_config.is_bgr()) {
    data_provider_image_option_.target_color = Color::BGR;
    image_mean_[0] = part1_config.mean_b();
    image_mean_[1] = part1_config.mean_g();
    image_mean_[2] = part1_config.mean_r();
    image_std_[0] = part1_config.std_b();
    image_std_[1] = part1_config.std_g();
    image_std_[2] = part1_config.std_r();
  } else {
    data_provider_image_option_.target_color = Color::RGB;
    image_mean_[0] = part1_config.mean_r();
    image_mean_[1] = part1_config.mean_g();
    image_mean_[2] = part1_config.mean_b();
    image_std_[0] = part1_config.std_r();
    image_std_[1] = part1_config.std_g();
    image_std_[2] = part1_config.std_b();
  }
  
  data_provider_image_option_.do_crop = false;
  data_provider_image_option_.crop_roi.x = input_offset_x_;
  data_provider_image_option_.crop_roi.y = input_offset_y_;
  data_provider_image_option_.crop_roi.height = crop_height_;
  data_provider_image_option_.crop_roi.width = crop_width_;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id_);
  AINFO << "GPU: " << prop.name;

  const auto net_param = part1_config.net_config();

  for (int i = 0; i < net_param.input_names_size(); ++i) {
    AINFO << net_param.input_names(i);
    input_names_.push_back(net_param.input_names(i)); 
  }
  for (int i = 0; i < net_param.output_names_size(); ++i) {
    AINFO << net_param.output_names(i);
    output_names_.push_back(net_param.output_names(i)); 
  }
  RTEngine::Init();
  view_tran.Init(60, 256, 704, 16);  
  return true;
}

bool BEVPart1::Detect(std::vector<cv::Mat>& images) {
  auto start = std::chrono::high_resolution_clock::now();

  float depth[6*59*16*44];
  float feat[6*80*16*44];


  for(int n = 0; n < 6; n++) {
    auto input_blob = RTEngine::get_blob(input_names_[0]);

    cv::Mat img = images[n];

    bev::DataProvider data_provider;
    bev::DataProvider::InitOptions dp_init_options;
    dp_init_options.image_height = img.rows;
    dp_init_options.image_width = img.cols;
    dp_init_options.device_id = 0;
    data_provider.Init(dp_init_options);
    data_provider.FillImageData(img.rows, img.cols, (const uint8_t*)(img.data), "bgr8");

    // use data provider to crop input image
    if (!data_provider.GetImage(data_provider_image_option_, &image_src_)) {
      return false;
    }
#if 0 //debug
    std::string img_name = "image_src_" + std::to_string(n) + ".jpg";
    cv::Mat output_image1(image_src_.rows(), image_src_.cols(), CV_8UC3,
                          cv::Scalar(0, 0, 0));
    memcpy(output_image1.data, image_src_.cpu_data(),
            image_src_.total() * sizeof(uint8_t));
    cv::imwrite(img_name, output_image1);
#endif

    // // resize the cropped image into network input blob
    ResizeGPU(
        image_src_, input_blob, static_cast<int>(crop_width_), 0,
        static_cast<float>(image_mean_[0]), static_cast<float>(image_mean_[1]),
        static_cast<float>(image_mean_[2]), false, image_std_[0], image_std_[1], image_std_[2]);

#if 0 //debug

    std::vector<unsigned long> npy_shape;
    std::vector<float> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    AINFO << "compare ranks_feat";
    npy::LoadArrayFromNumpy("../data/inputs.npy", npy_shape, is_fortran, npy_vector);


  float* input_data = input_blob->mutable_cpu_data(); 
  for(int i = 0; i < npy_vector.size(); i++)
  {
    if (frame->frame_id == 0){
        if(abs(input_data[i] - npy_vector[i]) > 0.1)
          AINFO << i << " " << input_data[i] - npy_vector[i];
    } 
  }
#endif

    ADEBUG << "resize gpu finish.";
    cudaDeviceSynchronize();
    RTEngine::Infer();
    cudaDeviceSynchronize();
    AINFO << "infer finish.";

    auto output_blob = RTEngine::get_blob(output_names_[0]);
    int out_b = output_blob->num();
    int out_c = output_blob->channels();
    int out_h = output_blob->height();
    int out_w = output_blob->width();
    float* output_data = output_blob->mutable_cpu_data(); 

    int depth_size = out_b  * 59 * out_h * out_w;
    // int feat_size = out_b  * 80 * out_h * out_w;
    // float depth[depth_size];
    // float feat[feat_size];
    for (int b = 0; b < out_b; b++) {
      for (int c = 0; c < out_c; c++) {
        for (int h = 0; h < out_h; h++) {
          for (int w = 0; w < out_w; w++) {
            int index = b * out_c * out_h * out_w + c * out_h * out_w + h * out_w + w;
            if (c < 59) 
              depth[n * 59*16*44 +  index] = output_data[index];
            else
              feat[n * 80*16*44 + (index - depth_size)] = output_data[index];
          }
        }
      }
    }
    //softmax(depth, 59, 16, 44);
#if 0
    std::vector<unsigned long> npy_shape;
    std::vector<float> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    npy::LoadArrayFromNumpy("../data/fea.npy", npy_shape, is_fortran, npy_vector);
    // AINFO <<frame->frame_id;
    if (frame->frame_id == 0) {
      for (int i = 0; i < 80; i++) {
        for (int j = 0; j < 16;j++) {
          for(int z = 0; z < 44; z++) {
            int ind = i * 44 * 16 + j*44 + z;
            float sub = feat[ind] - npy_vector[ind];
              AINFO << feat[ind] << " " << " " << npy_vector[ind] << " " << sub;
          } 
        }
      }
    }
#endif
  }
  softmax(depth, 6, 59, 16, 44);
  float feat_pose[ 6*80*16*44];
  tran_pose(feat, feat_pose, 6, 80, 16, 44);
#if 0
    std::vector<unsigned long> npy_shape;
    std::vector<float> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    npy::LoadArrayFromNumpy("../data/fea.npy", npy_shape, is_fortran, npy_vector);
    // AINFO <<frame->frame_id;
    // for (int n = 0; n < 6; n++) {
    //   for (int i = 0; i < 80; i++) {
    //     for (int j = 0; j < 16;j++) {
    //       for(int z = 0; z < 44; z++) {
    //         int ind = n*59*16*44 + i * 44 * 16 + j*44 + z;
    //         float sub = feat[ind] - npy_vector[ind];
    //           AINFO << feat[ind] << " " << " " << npy_vector[ind] << " " << sub;
    //       } 
    //     }
    //   }
    // }


    for(int i = 0; i < npy_vector.size(); i++) {
      float sub = feat_pose[i] - npy_vector[i];
      AINFO << feat_pose[i] << " " << " " << npy_vector[i] << " " << sub;
    }
#endif
  view_tran.BEVPool_V2(depth, feat_pose);
  return true;
}
}  // namespace bev
