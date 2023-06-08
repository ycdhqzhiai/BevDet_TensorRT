/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 11:17:59 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-07 17:02:18
 */
#include "bevdet/bev_part1.h"
#include "base/distortion_model.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    AINFO << "input model path";
    return -1;
  }

  std::string model_path = (std::string)argv[1];

  bev::BEVPart1 bev_part1;

  bev_part1.Init(model_path + "/bevdet.pt");

  std::vector<cv::Mat> images;
  for (int i = 0; i < 6; i++) {
    std::string img_path = model_path + "/batch_" + std::to_string(i) + ".jpg";
    cv::Mat img = cv::imread(img_path);

    if(img.empty())
      break;
    images.push_back(img);
    
  }
  bev_part1.Detect(images);


  return 0;
}