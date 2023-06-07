/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 10:40:51 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-06 16:03:37
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

#include "bevdet/utils/resize.h"
namespace bev {

class BEVPart1 : public RTEngine{
 public:
  BEVPart1(){
    src_height_ = 0;
    src_width_ = 0;
    input_offset_y_ = 0;
    input_offset_x_ = 0;
    crop_height_ = 0;
    crop_width_ = 0;
    resize_height_ = 0;
    resize_width_ = 0;
    image_mean_[0] = 0;
    image_mean_[1] = 0;
    image_mean_[2] = 0;
  }

  ~BEVPart1() {}

  bool Init(const std::string& config_path);

  // @brief: detect lane from image.
  // @param [in]: options
  // @param [in/out]: frame
  // detected lanes should be filled, required,
  // 3D information of lane can be filled, optional.
  bool Detect(CameraFrame *frame);
//   std::string Name();

//  private:
//  void GenerateAnchor();
//  // bool PostProcess();
//   void Pred2Coords(CameraFrame *frame);

//   std::shared_ptr<inference::Inference> cnnadapter_lane_ = nullptr;

  // parameters for data provider
  uint16_t src_height_;
  uint16_t src_width_;
  uint16_t input_offset_y_;
  uint16_t input_offset_x_;
  uint16_t crop_height_;
  uint16_t crop_width_;
  uint16_t resize_height_;
  uint16_t resize_width_;
  float image_mean_[3];
  float image_std_[3];
  int device_id_;
  std::vector<float> anchors_;

  DataProvider::ImageOptions data_provider_image_option_;
  Image8U image_src_;
  // std::vector<std::string> net_inputs_;
  // std::vector<std::string> net_outputs_;
  // std::vector<base::LaneLine> lane_objects_;

};

}  // namespace bev
