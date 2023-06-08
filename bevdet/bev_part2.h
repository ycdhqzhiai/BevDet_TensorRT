/*
 * @Author: ycdhq 
 * @Date: 2023-06-08 10:23:17 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 10:54:42
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

  uint16_t src_height_;
  uint16_t src_width_;

  int device_id_;
  std::vector<std::string> task_name = {"reg", "height", "dim", "rot", "heatmap"};

};

}  // namespace bev
