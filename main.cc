/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 11:17:59 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-13 16:21:19
 */
#include "bevdet/bev_part1.h"
#include "bevdet/bev_part2.h"
#include "base/sensor_param.h"
#include <opencv2/opencv.hpp>

std::string views[6] = {"cam_front_left", "cam_front", "cam_front_right",
                        "cam_back_left", "cam_back", "cam_back_right"};

std::string views_img[6] = {"../data/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151604004799.jpg",
                            "../data/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404.jpg",
                            "../data/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151604020482.jpg",
                            "../data/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151604047405.jpg",
                            "../data/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151604037558.jpg",
                            "../data/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151604028370.jpg"};

void draw_box(const std::vector<cv::Mat>& images, 
  std::vector<bev::NuscenesBox>& pre_boxs) {
  SensorParam* sensor_param = SensorParam::Instance();

  for (int i = 0; i < images.size(); i++) {
    cv::Mat show_img = cv::imread(views_img[i]);

    std::string view = views[i] + "2lidar";
    Eigen::Matrix4d lidar2camera = sensor_param->GetProjectionMatrix(view);

    Eigen::Matrix3f camera_intrinsic = sensor_param->intrinsic_map_[i];
   
    AINFO << camera_intrinsic;

    for(auto& pre_box : pre_boxs) {
      //cv::Point2f show_points[8];
      std::vector<cv::Point2f> show_points;
      for (int j = 0; j < 8; j++) {
        Eigen::Vector4d pts_camera = static_cast<Eigen::Matrix<double, 4, 1, 0, 4, 1>>(
              lidar2camera.inverse() * Eigen::Vector4d(pre_box.corner_x_[j],
                                          pre_box.corner_y_[j],
                                          pre_box.corner_z_[j], 1.0));
        Eigen::Vector3d pts_img = camera_intrinsic.cast<double>() * Eigen::Vector3d(pts_camera(0) / pts_camera(2),
                                          pts_camera(1) / pts_camera(2),
                                          1.0);
        
        if (pts_camera(2) > 0.5 && pts_img(0) > 0 &&
          pts_img(1) > 0 && pts_img(0) < show_img.cols &&
          pts_img(1) < show_img.rows) {
          show_points.push_back(cv::Point2f(pts_img(0), pts_img(1)));
        }
      }

      for (int n = 0; n < show_points.size(); n++) {
        for (int m = 0; m < show_points.size(); m++) {
          if (m != n)
            cv::line(show_img, show_points[n], show_points[m], cv::Scalar(0, 0, 255), 5, cv::LINE_8);
        }
      }
    }
    
    cv::imwrite("show_img.jpg", show_img);
    break;
  }
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    AINFO << "input model path";
    return -1;
  }

  std::string model_path = (std::string)argv[1];

  bev::BEVPart1 bev_part1;
  bev::BEVPart2 bev_part2;

  bev_part1.Init(model_path + "/bevdet.pt");
  bev_part2.Init(model_path + "/bevdet.pt");

  std::vector<cv::Mat> images;
  for (int i = 0; i < 6; i++) {
    std::string img_path = model_path + "/batch_" + std::to_string(i) + ".jpg";
    cv::Mat img = cv::imread(img_path);

    if(img.empty())
      break;
    images.push_back(img);
    
  }
  bev_part1.Detect(images);

  bev_part2.Detect(bev_part1.view_tran.bev_feature_);

  draw_box(images, bev_part2.pre_box_);


  return 0;
}