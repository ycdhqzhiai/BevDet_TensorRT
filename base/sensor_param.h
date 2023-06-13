/*
 * @Author: ycdhq 
 * @Date: 2023-04-14 16:40:44 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-13 10:24:36
 */
#pragma once

#include <string>
#include "Eigen/Eigen"
#include "Eigen/Core"
#include "base/macros_common.h"
#include <memory>
#include <mutex>
#include "common/log.h"
#include "base/eigen_defs.h"

class SensorParam {
  public:
    bool Init();
    bool LoadSensorParam(const std::string &yaml_file);
    Eigen::Matrix4d GetProjectionMatrix(const std::string& name) const;
  private:
    std::mutex mutex_;
    bool inited_ = false;

  public:    
    int camera_height_;
    int camera_width_;


    EigenVector<Eigen::Matrix3f> intrinsic_map_;
    EigenVector<Eigen::Matrix3f> rots_map_;
    EigenVector<Eigen::Vector3f> trans_map_;

    Eigen::Matrix3f post_rots_;
    Eigen::Vector3f post_trans_;

    EigenMap<std::string, Eigen::Matrix4d> coordinate_map_;
  DECLARE_SINGLETON(SensorParam);
};