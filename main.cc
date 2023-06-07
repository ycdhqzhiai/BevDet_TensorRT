/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 11:17:59 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 09:51:38
 */
#include "bevdet/bev_part1.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    AINFO << "input model path";
    return -1;
  }

  std::string model_path = (std::string)argv[1];

  bev::BEVPart1 bev_part1;
  bev_part1.Init(model_path + "/bevdet.pt");

  for (int i = 0; i < 6; i++) {
    std::string img_path = model_path + "/batch_" + std::to_string(i) + ".jpg";
    cv::Mat img = cv::imread(img_path);

    if(img.empty())
      break;

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
    bev_part1.Detect(&frame);
  }

  return 0;
}