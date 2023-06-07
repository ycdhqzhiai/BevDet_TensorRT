/*
 * @Author: ycdhq 
 * @Date: 2023-04-14 16:40:44 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 11:11:16
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

class BaseCameraDistortionModel {
  public:
    bool Init();
    bool LoadCameraIntrinsic(const std::string &yaml_file);

  private:
    bool set_params(size_t width, size_t height,
                  const Eigen::VectorXf& params);

  private:
    std::mutex mutex_;
    bool inited_ = false;

  public:    
    int camera_height_;
    int camera_width_;
    // EigenMap<std::string, Eigen::Matrix3f> intrinsic_map_;
    // EigenMap<std::string, Eigen::Matrix3f> rots_map_;
    // EigenMap<std::string, Eigen::Vector3f> trans_map_;
    EigenVector<Eigen::Matrix3f> intrinsic_map_;
    EigenVector<Eigen::Matrix3f> rots_map_;
    EigenVector<Eigen::Vector3f> trans_map_;

    Eigen::Matrix3f post_rots_;
    Eigen::Vector3f post_trans_;
  DECLARE_SINGLETON(BaseCameraDistortionModel);
};