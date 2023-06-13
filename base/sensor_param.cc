/*
 * @Author: ycdhq 
 * @Date: 2023-04-18 09:24:55 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-13 10:46:04
 */
#include <iostream>
#include <fstream>
#include "sensor_param.h"
#include "yaml-cpp/yaml.h"
#include "json/json.h"

SensorParam::SensorParam() {
  Init();
}

bool SensorParam::Init() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (inited_) {
    return true;
  }
  const std::string params_file = "../data/params.json";
  if (!LoadSensorParam(params_file))
    return false;
  inited_ = true;
  return true;
}

bool SensorParam::LoadSensorParam(const std::string &params_file) {
  std::ifstream ifs(params_file);
  Json::Value root;
  Json::Reader reader;  

  if(reader.parse(ifs,root)) {
    Json::Value &arr_cam = root["cam_param"];
    for (int n = 0; n < arr_cam.size(); n++) {
      Eigen::Matrix3f intrinsic;
      Eigen::Matrix3f rots;
      Eigen::Vector3f trans;
      for (int i = 0; i < arr_cam[n]["Intrinsics"].size(); i++) {
        for (int j = 0; j < arr_cam[n]["Intrinsics"][i].size(); j++) {
          intrinsic(i, j) = arr_cam[n]["Intrinsics"][i][j].asFloat();
        }
      }

      for (int i = 0; i < arr_cam[n]["Extrinsics"].size(); i++) {
        for (int j = 0; j < arr_cam[n]["Extrinsics"][i].size(); j++) {
          rots(i, j) = arr_cam[n]["Extrinsics"][i][j].asFloat();
        }
      }

      for (int i = 0; i < arr_cam[n]["Translation"].size(); i++) {
        trans[i] = arr_cam[n]["Translation"][i].asFloat();
      }
      // intrinsic_map_.insert(std::pair<std::string, Eigen::Matrix3f>(arr[n]["name"].asString(), intrinsic));
      // rots_map_.insert(std::pair<std::string, Eigen::Matrix3f>(arr[n]["name"].asString(), rots));
      // trans_map_.insert(std::pair<std::string, Eigen::Vector3f>(arr[n]["name"].asString(), trans));
      intrinsic_map_.push_back(intrinsic);
      rots_map_.push_back(rots);
      trans_map_.push_back(trans);
    }

    Json::Value &arr_coor = root["coordinate_param"];

    for (int n = 0; n < arr_coor.size(); n++) {
      Eigen::Matrix3f rotation;
      Eigen::Vector3f trans;
      for (int i = 0; i < arr_coor[n]["rotation"].size(); i++) {
        for (int j = 0; j < arr_coor[n]["rotation"][i].size(); j++) {
          rotation(i, j) = arr_coor[n]["rotation"][i][j].asFloat();
        }
      }
      
      for (int i = 0; i < arr_coor[n]["Translation"].size(); i++) {
        trans[i] = arr_coor[n]["Translation"][i].asFloat();
      }

      Eigen::Matrix4d projection_matrix;
      projection_matrix.setIdentity();
      projection_matrix.block<3, 3>(0, 0) = rotation.cast <double>();
      projection_matrix.topRightCorner<3, 1>() = trans.cast <double>();
      coordinate_map_.insert(std::pair<std::string, Eigen::Matrix4d>(arr_coor[n]["name"].asString(), projection_matrix));
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
  return true;
}

Eigen::Matrix4d SensorParam::GetProjectionMatrix(
    const std::string& name) const {
  Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();

  const auto& itr = coordinate_map_.find(name);

  return itr == coordinate_map_.end() ? identity : itr->second;
}
