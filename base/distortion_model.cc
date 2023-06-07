/*
 * @Author: ycdhq 
 * @Date: 2023-04-18 09:24:55 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 11:09:59
 */
#include <iostream>
#include <fstream>
#include "distortion_model.h"
#include "yaml-cpp/yaml.h"
#include "json/json.h"

BaseCameraDistortionModel::BaseCameraDistortionModel() {
  Init();
}

bool BaseCameraDistortionModel::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (inited_) {
    return true;
  }
  const std::string params_file = "../data/camera_params.json";
  if (!LoadCameraIntrinsic(params_file))
    return false;
  inited_ = true;
  return true;
}

bool BaseCameraDistortionModel::LoadCameraIntrinsic(const std::string &params_file) {
  std::ifstream ifs(params_file);
  Json::Value root;
  Json::Reader reader;  

  if(reader.parse(ifs,root)) {
    Json::Value &arr = root["cam_param"];
    for (int n = 0; n < arr.size(); n++) {
      Eigen::Matrix3f intrinsic;
      Eigen::Matrix3f rots;
      Eigen::Vector3f trans;
      for (int i = 0; i < arr[n]["Intrinsics"].size(); i++) {
        for (int j = 0; j < arr[n]["Intrinsics"][i].size(); j++) {
          intrinsic(i, j) = arr[n]["Intrinsics"][i][j].asFloat();
        }
      }

      for (int i = 0; i < arr[n]["Extrinsics"].size(); i++) {
        for (int j = 0; j < arr[n]["Extrinsics"][i].size(); j++) {
          rots(i, j) = arr[n]["Extrinsics"][i][j].asFloat();
        }
      }

      for (int i = 0; i < arr[n]["Translation"].size(); i++) {
        trans[i] = arr[n]["Translation"][i].asFloat();
      }
      // intrinsic_map_.insert(std::pair<std::string, Eigen::Matrix3f>(arr[n]["name"].asString(), intrinsic));
      // rots_map_.insert(std::pair<std::string, Eigen::Matrix3f>(arr[n]["name"].asString(), rots));
      // trans_map_.insert(std::pair<std::string, Eigen::Vector3f>(arr[n]["name"].asString(), trans));
      intrinsic_map_.push_back(intrinsic);
      rots_map_.push_back(rots);
      trans_map_.push_back(trans);
    }
  } else {
    AERROR << params_file << " parse failed!";
    return false;
  }

  post_rots_(0, 0) = 0.44;
  post_rots_(0, 1) = 0;
  post_rots_(0, 2) = 0;
  post_rots_(1, 0) = 0;
  post_rots_(1, 1) = 0.44;
  post_rots_(1, 2) = 0;
  post_rots_(2, 0) = 0;
  post_rots_(2, 1) = 0;
  post_rots_(2, 2) = 1;

  post_trans_[0] = 0;
  post_trans_[1] = -140.0;
  post_trans_[2] = 0;


  // YAML::Node node = YAML::LoadFile(params_file);
  // if (node.IsNull()) {
  //   AINFO << params_file << " can not be load";
  //   return false;
  // }

  // float camera_width = 0.0f;
  // float camera_height = 0.0f;
  // try {
  //   camera_width = node["width"].as<float>();
  //   camera_height = node["height"].as<float>();
  //   for (size_t i = 0; i < 9; ++i) {
  //     params(i) = node["K"][i].as<float>();
  //   }
  //   for (size_t i = 0; i < 5; ++i) {
  //     params(9 + i) = node["D"][i].as<float>();
  //   }

  //   params(9 + 5 + 0) = node["transform"]["rotation"]["w"].as<float>();
  //   params(9 + 5 + 1) = node["transform"]["rotation"]["x"].as<float>();
  //   params(9 + 5 + 2) = node["transform"]["rotation"]["y"].as<float>();
  //   params(9 + 5 + 3) = node["transform"]["rotation"]["z"].as<float>();
  //   params(9 + 5 + 4) = node["transform"]["translation"]["x"].as<float>();
  //   params(9 + 5 + 5) = node["transform"]["translation"]["y"].as<float>();
  //   params(9 + 5 + 6) = node["transform"]["translation"]["z"].as<float>();

  //   set_params(static_cast<size_t>(camera_width),
  //                     static_cast<size_t>(camera_height), params);
  // } catch (YAML::Exception &e) {
  //   std::cout << "load camera intrisic file " << params_file
  //          << " with error, YAML exception: " << e.what();
  //   return false;
  // }

  return true;
}

// bool BaseCameraDistortionModel::set_params(size_t width, size_t height,
//                                             const Eigen::VectorXf& params) {
//   if (params.size() != 21) {
//     return false;
//   }
//   camera_width_ = width;
//   camera_height_ = height;
//   intrinsic_params_(0, 0) = params(0);
//   intrinsic_params_(0, 1) = params(1);
//   intrinsic_params_(0, 2) = params(2);
//   intrinsic_params_(1, 0) = params(3);
//   intrinsic_params_(1, 1) = params(4);
//   intrinsic_params_(1, 2) = params(5);
//   intrinsic_params_(2, 0) = params(6);
//   intrinsic_params_(2, 1) = params(7);
//   intrinsic_params_(2, 2) = params(8);
//   distort_params_[0] = params[9];
//   distort_params_[1] = params[10];
//   distort_params_[2] = params[11];
//   distort_params_[3] = params[12];
//   distort_params_[4] = params[13];

//   Eigen::Quaterniond q;
//   q.x() = params(14);
//   q.y() = params(15);
//   q.z() = params(16);
//   q.w() = params(17);
//   rots_ = q.normalized().toRotationMatrix();
//   trans_[0] = params[18];
//   trans_[1] = params[19];
//   trans_[2] = params[20];

//   post_rots_(0, 0) = 0.44;
//   post_rots_(0, 1) = 0;
//   post_rots_(0, 2) = 0;
//   post_rots_(1, 0) = 0;
//   post_rots_(1, 1) = 0.44;
//   post_rots_(1, 2) = 0;
//   post_rots_(2, 0) = 0;
//   post_rots_(2, 1) = 0;
//   post_rots_(2, 2) = 1;

//   post_trans_[0] = 0;
//   post_trans_[1] = -140.0;
//   post_trans_[2] = 0;

//   return true;
// }