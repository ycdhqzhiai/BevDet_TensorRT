/*
 * @Author: ycdhq 
 * @Date: 2023-06-09 15:25:48 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-13 15:37:23
 */


#include "nuscenes_box.h"
#include "common/log.h"
#include "base/sensor_param.h"
#include "common/math.h"
namespace bev {

NuscenesBox::NuscenesBox(const bev::BBox& res) {

  center_[0] = res.bbox[0];
  center_[1] = res.bbox[1];
  center_[2] = res.bbox[2];

  wlh_[0] = res.bbox[4];
  wlh_[1] = res.bbox[3];
  wlh_[2] = res.bbox[5];
  yaw_ = res.bbox[6];

  score_ = res.score;
  label_ = res.label;
  Corners();
}

void NuscenesBox::Corners(){
  SensorParam* sensor_param = SensorParam::Instance();
  
  Eigen::Matrix4d eog2global = sensor_param->GetProjectionMatrix("eog2global");
  Eigen::Matrix4d lidar2eog = sensor_param->GetProjectionMatrix("lidar2ego");
  Eigen::Matrix4d lidar2global = eog2global * lidar2eog;

  Eigen::Vector4d pts_center = static_cast<Eigen::Matrix<double, 4, 1, 0, 4, 1>>(
        eog2global * Eigen::Vector4d(center_[0],
                                     center_[1],
                                     center_[2], 1.0));

  // Eigen::Matrix3d rot = trans_matix.block<3, 3>(0, 0);

  global_center_[0] = pts_center(0);
  global_center_[1] = pts_center(1);
  global_center_[2] = pts_center(2);
  //AINFO << pts_center;
  global_rot_ = Eigen::AngleAxisd(yaw_, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  global_orientation_ = eog2global.block<3, 3>(0, 0) * global_rot_;

  Eigen::Vector3d euler_angles = global_orientation_.eulerAngles(0, 1, 2); 

  float lidar_yaw = euler_angles(2) + M_PI_2;
  Eigen::Matrix3d lidar_rot = Eigen::AngleAxisd(lidar_yaw, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();;



  for (int i = 0; i < 8; i++) {
    corner_x_[i] = (corner_x_[i] - origin_[0]) * wlh_[0];
    corner_y_[i] = (corner_y_[i] - origin_[1]) * wlh_[1];
    corner_z_[i] = (corner_z_[i] - origin_[2]) * wlh_[2];
    Eigen::Matrix<float, 3, 1> points(corner_x_[i], corner_y_[i], corner_z_[i]);
    Eigen::Matrix<float, 3, 1> new_point = lidar_rot.cast <float>() * points;    
    corner_x_[i] = new_point(0) + global_center_[0];
    corner_y_[i] = new_point(1) + global_center_[1];
    corner_z_[i] = new_point(2) + global_center_[2];

    Eigen::Vector4d pts_lidar = static_cast<Eigen::Matrix<double, 4, 1, 0, 4, 1>>(
          lidar2global.inverse() * Eigen::Vector4d(corner_x_[i], corner_y_[i], corner_z_[i], 1.0));

    corner_x_[i] = pts_lidar(0);
    corner_y_[i] = pts_lidar(1);
    corner_z_[i] = pts_lidar(2);
  }
}




} //namespace bev