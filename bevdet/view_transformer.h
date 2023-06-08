/*
 * @Author: ycdhq 
 * @Date: 2023-04-14 14:29:57 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 10:11:30
 */

#pragma once
#include <iostream>

#include <Eigen/Dense>
#include "unsupported/Eigen/CXX11/Tensor"
#include "base/distortion_model.h"
#include "common/npy.h"
#include "common/log.h"
#include "bevdet/view_transformer.h"
class LSSViewTransformer {
  public:
    
    LSSViewTransformer();
    ~LSSViewTransformer();
    void Init(int depth, int H_in, int W_in, int downsample);
    void BEVPool_V2(float* dep, float* fea);
    // void BEVPool_V2();

  private:
    void LoadFeatureMapData();
    void CreateFrustumAndLidarPoint();
    void VoxelPoolingPrepare();
    void CheckData(std::vector<float>& dep, std::vector<float>& fea);

    int D_;
    int N_ = 6;
    int H_in_;
    int W_in_;
    int downsample_;
    Eigen::Vector3f grid_lower_bound_;
    Eigen::Vector3f grid_interval_;
    Eigen::Vector3f grid_size_;

    int W_feat_;
    int H_feat_;
    Eigen::Tensor<float, 5> frustum_;

    BaseCameraDistortionModel *camera_model_ = nullptr;

public:

    std::vector<unsigned long> feature_shape_;
    std::vector<unsigned long> depth_shape_;

    std::vector<float> feature_map_; // 必须指定<dtype>类型与npy对应
    std::vector<float> depth_map_; // 必须指定<dtype>类型与npy对应    

    std::vector<int> ranks_bev_out_;
    std::vector<int> ranks_feat_out_;
    std::vector<int> ranks_depth_out_;    
    std::vector<int> interval_starts_out_;
    std::vector<int> interval_lengths_out_;
    // std::vector<float> bev_feature_;
    //float* bev_feature_;
    float* bev_feature_;
    // std::unique_ptr<pcl::PointCloud<pcl::PointXYZ>> CloudT_;

}; //class LSSViewTransformer
