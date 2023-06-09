/*
 * @Author: ycdhq 
 * @Date: 2023-06-08 10:23:17 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-09 12:03:08
 */

#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

#include "base/camera_frame.h"
#include "base/data_provider.h"
#include "proto/bevdet.pb.h"
#include "rt_engine.h"
#include "base/eigen_defs.h"
#include "bevdet/utils/resize.h"
#include "bevdet/view_transformer.h"
namespace bev {
using out_tensor = std::pair<int, float*>;

struct CenterPointResult {
  /// Bounding box 3d: {x, y, z, x_size, y_size, z_size, yaw,vel1,vel2}
  // float bbox[9];
  /// Score
  float bbox[7];

  float score;
  /// the class label
  //'car',         'truck',   'construction_vehicle',
  //'bus',         'trailer', 'barrier',
  //'motorcycle',  'bicycle', 'pedestrian',
  //'traffic_cone'
  uint32_t label;
};

class BEVPart2 : public RTEngine {
 public:
  BEVPart2(){
    src_height_ = 0;
    src_width_ = 0;
  }

  ~BEVPart2() {}

  bool Init(const std::string& config_path);

  // @brief: detect lane from image.
  // @param [in]: options
  // @param [in/out]: frame
  // detected lanes should be filled, required,
  // 3D information of lane can be filled, optional.
  bool Detect(float* bev_feature);

private:  
  int decode(const out_tensor& regtb,
                    const out_tensor& heitb,
                    const out_tensor& dimtb,
                    const out_tensor& rottb,
                    const out_tensor& heatmaptb, float score_threshold,
                    uint32_t& first_label, std::vector<CenterPointResult>& res);

  uint16_t src_height_;
  uint16_t src_width_;

  int device_id_;
  std::vector<std::string> task_name = {"reg", "height", "dim", "rot", "heatmap"};

  int out_size_factor_ = 8;
  float voxel_size_[2] = {0.1, 0.1};
  float pc_range_[2] = {-51.2, -51.2};
  float post_center_range_[6] = {-61.2, -61.2, -10.0, 61.2, 61.2, 10.0};
  int post_max_size_ = 500;
  float nms_thr_[6] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
  float factor_[6] = {1.0,0.7,0.4,1.0,1.0,4.5};
};

}  // namespace bev
