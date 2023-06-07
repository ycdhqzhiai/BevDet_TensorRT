/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 11:17:59 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 12:06:05
 */
#include "bevdet/bev_part1.h"
#include "base/distortion_model.h"
#include <opencv2/opencv.hpp>
#include "bevdet/view_transformer.h"
int main(int argc, char* argv[]) {
  if (argc < 2) {
    AINFO << "input model path";
    return -1;
  }

  std::string model_path = (std::string)argv[1];

  bev::BEVPart1 bev_part1;
  LSSViewTransformer view_tran;

  bev_part1.Init(model_path + "/bevdet.pt");
  view_tran.Init(60, 256, 704, 16);
  BaseCameraDistortionModel* camera_model = BaseCameraDistortionModel::Instance();

  for (int i = 0; i < 6; i++) {
    std::string img_path = model_path + "/batch_" + std::to_string(i) + ".jpg";
    cv::Mat img = cv::imread(img_path);

    if(img.empty())
      break;
    Eigen::Matrix3f camera_intrinsic = camera_model->intrinsic_map_[i];
    AINFO << camera_intrinsic;
    cv::Mat draw_mat = img.clone();
    bev::CameraFrame frame;
    bev::DataProvider data_provider;
    frame.data_provider = &data_provider;
    bev::DataProvider::InitOptions dp_init_options;
    dp_init_options.image_height = img.rows;
    dp_init_options.image_width = img.cols;
    dp_init_options.device_id = 0;
    frame.data_provider->Init(dp_init_options);
    frame.data_provider->FillImageData(img.rows, img.cols, (const uint8_t*)(img.data), "bgr8");
    frame.frame_id = i;
    frame.camera_k_matrix = camera_model->intrinsic_map_[i];
    frame.camera_rot_matrix = camera_model->rots_map_[i];
    frame.camera_trans_vector = camera_model->trans_map_[i];
    bev_part1.Detect(&frame);
  }

  return 0;
}