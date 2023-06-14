

/*
 * @Author: ycdhq 
 * @Date: 2023-06-09 14:44:18 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-14 16:17:02
 */
#pragma once

#include <Eigen/Dense>

namespace bev {


struct BBox {
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
  bool isDrop = false;
};

class NuscenesBox {
public:
  NuscenesBox(const bev::BBox& res);
  ~NuscenesBox() {}
  void Corners();


  float center_[3];
  float global_center_[3];
  float wlh_[3];
  float yaw_;
  u_int32_t label_;
  float score_;
  Eigen::Matrix3d global_orientation_;
  Eigen::Matrix3d global_rot_;
  float corner_x_[8] = {0,0,0,0,1,1,1,1};
  float corner_y_[8] = {0,0,1,1,0,0,1,1};
  float corner_z_[8] = {0,1,1,0,0,1,1,0};

private:
  float origin_[3] = {0.5, 0.5, 0};


};
} // namespace bev