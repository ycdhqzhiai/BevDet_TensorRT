/*
 * @Author: ycdhq 
 * @Date: 2023-04-14 14:38:20 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 11:01:24
 */
#include "view_transformer.h"

bool cmp(int v1,int v2)
{
	return v1 < v2;
}

template <typename T1>
static std::vector<T1> tran_pose(std::vector<T1>& input, int N, int C, int H, int W){
                        
  std::vector<T1> out;
  out.resize(input.size());

//   for(int n = 0; n < N; n++){
//     for(int c = 0; c < C; c++) {
//       for(int h = 0; h < H; h++) {
//         for(int w = 0; w < W; w++) {
//           int src_index = n * C * H * W + c * H * W + h * W + w;
//           int dst_index = n * H * W * C + h * W * C + w * C + c;
//           out[dst_index] = input[src_index];
//         }
//       }
//     }
//   }

  for(int n = 0; n < N; n++){
    for(int h = 0; h < H; h++) {
      for(int w = 0; w < W; w++) {
        for(int c = 0; c < C; c++) {
          int dst_index = n * C * H * W + c * H * W + h * W + w;
          int src_index = n * H * W * C + h * W * C + w * C + c;
          out[dst_index] = input[src_index];
        }
      }
    }
  }

  return out;
}

LSSViewTransformer::LSSViewTransformer() {
    feature_map_.clear();
    depth_map_.clear();   
    ranks_bev_out_.clear();
    ranks_feat_out_.clear();
    ranks_depth_out_.clear();    
    interval_starts_out_.clear();
    interval_lengths_out_.clear();
}


LSSViewTransformer::~LSSViewTransformer() {}

void LSSViewTransformer::Init(int depth, int H_in, int W_in, int downsample) {
    D_ = depth;
    H_in_ = H_in;
    W_in_ = W_in;
    downsample_ = downsample;

    H_feat_ = H_in / downsample;
    W_feat_ = W_in / downsample;
    AINFO << "D " << D_ - 1 << " H_feat " << W_feat_ << " W_feat " << W_feat_; // output: 1

    camera_model_ = BaseCameraDistortionModel::Instance();


    Eigen::Vector3f x(-51.2, 51.2, 0.8);
    Eigen::Vector3f y(-51.2, 51.2, 0.8);
    Eigen::Vector3f z(-5.0, 3, 8);
    grid_lower_bound_[0] = x[0];
    grid_lower_bound_[1] = y[0];
    grid_lower_bound_[2] = z[0];
    grid_interval_[0] = x[2];
    grid_interval_[1] = y[2];
    grid_interval_[2] = z[2];
    grid_size_[0] = (x[1] - x[0]) / x[2];
    grid_size_[1] = (y[1] - y[0]) / y[2];
    grid_size_[2] = (z[1] - z[0]) / z[2];
    frustum_ = Eigen::Tensor<float, 5>(N_, D_ -1, H_feat_, W_feat_, 3);
    CreateFrustumAndLidarPoint();
    //LoadFeatureMapData();
    VoxelPoolingPrepare();
}

void LSSViewTransformer::LoadFeatureMapData() {
    const std::string feature_map_file = "npy_data/feat.npy";
    const std::string depth_map_file = "npy_data/depth.npy";

    bool is_fortran;
    // load ndarray voxel as vector<float>
    npy::LoadArrayFromNumpy(feature_map_file, feature_shape_, is_fortran, feature_map_);
    npy::LoadArrayFromNumpy(depth_map_file, depth_shape_, is_fortran, depth_map_);

}

void LSSViewTransformer::CreateFrustumAndLidarPoint() {

    float x_offet = (W_in_ - 1) / float(W_feat_ - 1);
    float y_offet = (H_in_ - 1) / float(H_feat_ - 1);

    for (int n = 0; n < N_; n++) {
        AINFO << "intrinsic "  << camera_model_->intrinsic_map_[n];
        Eigen::Matrix3f combine = camera_model_->rots_map_[n] * camera_model_->intrinsic_map_[n].inverse();
        for (int d =0; d < D_ - 1; d++) {
            for (int u = 0; u < H_feat_; u++) {
                for (int v = 0; v < W_feat_; v++) {
                    // frustum_(n,d,u,v,0) = v * x_offet;
                    // frustum_(n,d,u,v,1) = u * y_offet + 140;
                    // frustum_(n,d,u,v,2) = d + 1;
                    float x = v * x_offet;
                    float y = u * y_offet + 140;
                    float z = d + 1;
                    Eigen::Matrix<float, 3, 1> points(x, y, z);
                    Eigen::Matrix<float, 3, 1> trans_point = camera_model_->post_rots_.inverse() * points;
                    Eigen::Matrix<float, 3, 1> normal_poin(trans_point(0) * trans_point(2),
                                                           trans_point(1) * trans_point(2),
                                                           trans_point(2));
                    Eigen::Matrix<float, 3, 1> bev_point = combine *  normal_poin;
                    frustum_(n,d,u,v,0) = ((bev_point(0) + camera_model_->trans_map_[n][0]) - grid_lower_bound_[0]) / grid_interval_[0];
                    frustum_(n,d,u,v,1) = ((bev_point(1) + camera_model_->trans_map_[n][1]) - grid_lower_bound_[1]) / grid_interval_[1];
                    frustum_(n,d,u,v,2) = ((bev_point(2) + camera_model_->trans_map_[n][2]) - grid_lower_bound_[2]) / grid_interval_[2];
                }
            }
        }
    }
}

void LSSViewTransformer::VoxelPoolingPrepare() {
    int num_points = N_ * (D_ - 1) * H_feat_ * W_feat_;
    Eigen::Tensor<int, 2> coord(num_points, 4);
    // int x = 1;

    std::vector<int> kept;
    std::vector<int> ranks_range(num_points,0);
    for (int n = 0; n < N_; n++) {
        for (int d =0; d < D_ - 1; d++) {
            for (int u = 0; u < H_feat_; u++) {
                for (int v = 0; v < W_feat_; v++) {
                    int index = n * (D_ - 1) * H_feat_ * W_feat_ + d * H_feat_ * W_feat_ + u * W_feat_ + v;
                    coord(index, 0) = int(frustum_(n,d,u,v,0));
                    coord(index, 1) = int(frustum_(n,d,u,v,1));
                    coord(index, 2) = int(frustum_(n,d,u,v,2));
                    coord(index, 3) = 0; 

                    if (coord(index, 0) >= 0 && coord(index, 0) < grid_size_[0] &&
                        coord(index, 1) >= 0 && coord(index, 1) < grid_size_[1] &&
                        coord(index, 2) >= 0 && coord(index, 2) < grid_size_[2])
                        kept.push_back(index);
                    // if (index == 704)
                    //     AINFO << n << " " << u << " " << v;
                    ranks_range[index] = n * H_feat_ * W_feat_ + u * W_feat_ + v;
                }
            }
        }
    }

#if 0

    std::vector<unsigned long> npy_shape;
    std::vector<int> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    AINFO << "compare ranks_feat";
    npy::LoadArrayFromNumpy("../data/ranks_feat_range.npy", npy_shape, is_fortran, npy_vector);
    AINFO << npy_vector.size();
    AINFO << ranks_range.size();
    // for (int i = 41536 - 100; i < 41536 + 100; i++) {
        //AINFO << i << " " << npy_vector[i] <<  " VS " << ranks_range[i];
    for (int i =0; i < ranks_range.size(); i++) {        
        int v = npy_vector[i] - ranks_range[i];
        if (v != 0) {
            AINFO << i;
            AINFO << v;
            break;
        }
    }
    return;    
#endif    
    
    int num_in = kept.size();
    Eigen::Tensor<int, 2> coord_in(num_in, 4);
    std::vector<int> ranks_bev(num_in, 0);
    std::vector<int> ranks_feat(num_in,0);

    for (int i = 0; i < num_in; i++) {
        ranks_bev[i] = coord(kept[i], 3) * grid_size_[2] * grid_size_[1] * grid_size_[0] +
                    coord(kept[i], 2) * (grid_size_[1] * grid_size_[0]) +
                    coord(kept[i], 1) * grid_size_[0] + coord(kept[i], 0);
        ranks_feat[i] = ranks_range[kept[i]];
    }

#if 0

    std::vector<unsigned long> npy_shape;
    std::vector<int> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    AINFO << "compare ranks_feat";
    npy::LoadArrayFromNumpy("../data/ranks_feat_kept.npy", npy_shape, is_fortran, npy_vector);
    for (int i = 0; i < npy_vector.size(); i++) {
        int v = npy_vector[i] - ranks_feat[i];
        if (v != 0) {
            AINFO << i;
            AINFO << v;
        }
    }
    return;
#endif    
    std::vector<int> order(num_in, 0);
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(),
            [&ranks_bev](size_t index_1, size_t index_2) { return ranks_bev[index_1] < ranks_bev[index_2]; });

    ranks_bev_out_ = std::vector<int>(num_in, 0);
    ranks_depth_out_ = std::vector<int>(num_in, 0);
    ranks_feat_out_ = std::vector<int>(num_in, 0);
    // interval_starts_out_ = std::vector<int>(num_in, 0);

    for (int i = 0; i < num_in; i++) {
        ranks_bev_out_[i] = ranks_bev[order[i]];
        ranks_depth_out_[i] = kept[order[i]];
        ranks_feat_out_[i] = ranks_feat[order[i]]; 
    }   



    interval_starts_out_.push_back(0);
    for (int i = 0; i < num_in - 1; i++) {
        if (ranks_bev_out_[i + 1] != ranks_bev_out_[i])
            interval_starts_out_.push_back(i+1);
    }

    int num_inter = interval_starts_out_.size();
    interval_lengths_out_ = std::vector<int>(num_inter, 0);

    for (int i = 0; i < num_inter - 1; i++) {
        interval_lengths_out_[i] = interval_starts_out_[i + 1] - interval_starts_out_[i];
    }
    interval_lengths_out_[num_inter - 1] = num_in - interval_starts_out_[num_inter - 1];
}

void LSSViewTransformer::CheckData(std::vector<float>& dep, std::vector<float>& fea) {
    std::string npy_file = "npy_data/interval_lengths.npy";

    bool is_fortran;
    // load ndarray voxel as vector<float>
    std::vector<unsigned long> shape;
    std::vector<int> data;
    AINFO << "Load ";

    npy::LoadArrayFromNumpy(npy_file, shape, is_fortran, data);
    std::cout << "npy size " << data.size() << std::endl;
    std::cout << "C++ size " << interval_lengths_out_.size() << std::endl;
    for (int i = 0 ; i < interval_lengths_out_.size(); i++) {
        // dep[i] = data[i];
        if ((data[i] - interval_lengths_out_[i]) != 0)
            std::cout <<  i << " " << data[i] << " " << interval_lengths_out_[i] << std::endl;
    }


    // npy_file = "npy_data/feat.npy";
    // shape.clear();
    // data.clear();

    // npy::LoadArrayFromNumpy(npy_file, shape, is_fortran, data);
    // std::cout << "npy size " << data.size() << std::endl;
    // std::cout << "C++ size " << fea.size() << std::endl;
    // for (int i = 0 ; i < data.size(); i++) {
    //     fea[i] = data[i];
    //     // if ((data[i] - dep[i]) != 0)
    //     //     std::cout <<  i << " " << data[i] << " " << dep[i] << std::endl;
    // }

}

void LSSViewTransformer::BEVPool_V2(std::vector<float>& dep, std::vector<float>& fea) {
    AINFO << dep.size() << " " << fea.size();

    int c = 80;
    int n_intervals = interval_lengths_out_.size();
    std::vector<float> out(128*128*80, 0);

    for(int idx = 0; idx < n_intervals * c; idx++) {
        int index = idx / c;
        int cur_c = idx % c;

        if (index >= n_intervals) continue;
        int interval_start = interval_starts_out_[index];
        int interval_length = interval_lengths_out_[index];
        float psum = 0;

        float cur_depth[interval_length] =  {0};   
        float cur_feat[interval_length] =  {0};        

        for(int i = 0; i < interval_length; i++){
            int val_index = interval_start + i;
            cur_depth[i] = dep[ranks_depth_out_[interval_start + i]];
            cur_feat[i] = fea[ranks_feat_out_[interval_start + i] * c + cur_c];            
            psum += cur_feat[i] * cur_depth[i];
        }
        int cur_rank = ranks_bev_out_[interval_start];
        out[cur_rank * c + cur_c] = (float)psum * 1.0;
    }

    bev_feature_ = tran_pose(out, 1, 80, 128, 128);
    for (int zz = 0; zz < 80; zz++) {
        // std::cout << out[i] << std::endl;
        AINFO << bev_feature_[zz];
    }    
#if 0

    std::vector<unsigned long> npy_shape;
    std::vector<float> npy_vector;

    bool is_fortran;
    // load ndarray voxel as vector<float>
    AINFO << "compare output";
    npy::LoadArrayFromNumpy("../data/out.npy", npy_shape, is_fortran, npy_vector);
    AINFO << npy_vector.size();

    for (int i = 0; i < npy_vector.size(); i++) {
        // AINFO << npy_vector[i] << " " << out[i];//; * 0.5672;
        // int v = npy_vector[i] - out[i];
        // if (v != 0) {
        //     AINFO << i;
        //     AINFO << v;
        // }
        if (npy_vector[i] != 0 && out[i] != 0)
            AINFO << npy_vector[i] / out[i];
    }
#endif    
}
