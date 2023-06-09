/*
 * @Author: ycdhq 
 * @Date: 2023-06-08 10:25:02 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-09 12:05:00
 */

#include "bev_part2.h"

#include <algorithm>
#include <map>
#include <cmath>
#include "common/file.h"
#include "common/npy.h"
#include "common/math.h"
#include "bevdet/utils.h"
constexpr int K = 500;

namespace bev {

using apollo::cyber::common::GetProtoFromFile;
using apollo::cyber::common::GetAbsolutePath;

bool BEVPart2::Init(const std::string& config_path) {
  BEVDet configs;

  AINFO << config_path;
  if (!GetProtoFromFile(config_path, &configs)) {
    return false;
  }

  auto part2_config = configs.part2_config();

  engine_file_ = part2_config.engine_file(); 
  // compute image provider parameters
  src_height_ = static_cast<uint16_t>(part2_config.src_height());
  src_width_ = static_cast<uint16_t>(part2_config.src_width());


  device_id_ = part2_config.gpu_id();

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id_);
  AINFO << "GPU: " << prop.name;

  const auto net_param = part2_config.net_config();

  for (int i = 0; i < net_param.input_names_size(); ++i) {
    AINFO << net_param.input_names(i);
    input_names_.push_back(net_param.input_names(i)); 
  }
  for (int i = 0; i < net_param.output_names_size(); ++i) {
    AINFO << net_param.output_names(i);
    output_names_.push_back(net_param.output_names(i)); 
  }
  RTEngine::Init();
  return true;
}

bool BEVPart2::Detect(float* bev_feature) {
  auto start = std::chrono::high_resolution_clock::now();

  auto input_blob = RTEngine::get_blob(input_names_[0]);

  memcpy(input_blob->mutable_cpu_data(), bev_feature, 80*128*128 * sizeof(float));

  ADEBUG << "resize gpu finish.";
  cudaDeviceSynchronize();
  RTEngine::Infer();
  cudaDeviceSynchronize();
  AINFO << "part2 infer finish.";

  uint32_t first_label = 0;

  std::vector<CenterPointResult> ress;

  for (int t = 0; t < 6; t++){

    std::vector<out_tensor>  preds_vecs;
    std::vector<CenterPointResult> res;

    for (int i = 0; i < 5; i++) {
      int index = t * 5 + i;
      std::string out_name = output_names_[index];
      auto output_blob = RTEngine::get_blob(output_names_[index]);
      int out_b = output_blob->num();
      int out_c = output_blob->channels();
      int out_h = output_blob->height();
      int out_w = output_blob->width();
      float* output_data = output_blob->mutable_cpu_data();
      preds_vecs.push_back(std::make_pair(out_c, output_data));
    }
    int decode_res_num = decode(preds_vecs[0],
                                preds_vecs[1],
                                preds_vecs[2],
                                preds_vecs[3],
                                preds_vecs[4],
                                0.1, first_label, res);
    if (decode_res_num > 0) {
      float bboxes[500][5];
      float box_areas[500];
      float for_box2d[500][4];
      Point box_corners[500][5];

      for (auto i = 0; i < decode_res_num; ++i) {
        xywhr2xyxyr(res[i].bbox, bboxes[i], factor_[t]);
      }
      std::vector<bool> exist_box(decode_res_num, true);
      std::vector<bool> is_first(decode_res_num, true);
      for (int i = 0; i < decode_res_num; ++i) {
        if (!exist_box[i]) continue;
        ress.push_back(res[i]);  // add a box as result
        for (int j = i + 1; j < decode_res_num; ++j) {
          if (!exist_box[j]) continue;
          float iou = iou_bev(is_first, box_areas, box_corners, for_box2d, bboxes, j, i);
          if (iou > nms_thr_[t])
            exist_box[j] = false;
        }
      }
    }
    AINFO << "ress num " << ress.size();
  }
  return true;
}

int BEVPart2::decode(const out_tensor& regtb,
                  const out_tensor& heitb,
                  const out_tensor& dimtb,
                  const out_tensor& rottb,
                  const out_tensor& heatmaptb, float score_threshold,
                  uint32_t& first_label, std::vector<CenterPointResult>& res) {
  float* heatmap = heatmaptb.second;
  int heatmap_shape = heatmaptb.first;

  float heatmap_sigmoid[heatmap_shape * 128 * 128];
  sigmoid_on_tensor(heatmap, heatmap_sigmoid, heatmap_shape * 128 * 128);


  float* reg = regtb.second;
  int reg_shape = regtb.first;

  float* hei = heitb.second;
  int hei_shape = heitb.first;

  float* dim = dimtb.second;
  int dim_shape = dimtb.first;

  float* rot = rottb.second;
  int rot_shape = rottb.first;

  int cat = heatmap_shape;

  auto w = 128;
  std::vector<float> top_scores;
  std::vector<uint32_t> top_inds;
  std::vector<uint32_t> top_clses;
  std::vector<uint32_t> top_ys;
  std::vector<uint32_t> top_xs;
  if (cat == 1) {
    std::tie(top_scores, top_inds) = topK(heatmap_sigmoid, heatmap_shape * 128 * 128, K, 0.1);
    top_clses = std::vector<uint32_t>(top_scores.size(), 0);
    top_ys = division_n(top_inds, w);
    top_xs = remainder_n(top_inds, w);
  } else if (cat == 2) {
    std::vector<float> topk_scores;
    std::vector<uint32_t> topk_inds;
    std::tie(topk_scores, topk_inds) =
        topK(heatmap_sigmoid, 128 * 128, 0, K, 0.1);

    auto cls_flag = topk_inds.size();
    auto temp = topK(heatmap_sigmoid, 128 * 128, 1, K, 0.1);

    topk_scores.insert(topk_scores.end(), temp.first.begin(), temp.first.end());
    topk_inds.insert(topk_inds.end(), temp.second.begin(), temp.second.end());
    std::vector<uint32_t> topk_ind;
    std::tie(top_scores, topk_ind) = topK(topk_scores, K, 0.1);

    top_inds = gather_feat(topk_inds, topk_ind);     
    for (auto&& i : topk_ind) {
      if (i < cls_flag)
        top_clses.push_back(0);
      else
        top_clses.push_back(1);
    }
    top_ys = gather_feat(division_n(topk_inds, w), topk_ind);
    top_xs = gather_feat(remainder_n(topk_inds, w), topk_ind);
  } else {
    AERROR << "The number of cat is not supported to be " << cat << std::endl;
  }

  int res_num = 0;

  if (top_scores.size() > 0) {
    auto reg_trans = reshape(reg, reg_shape);
    auto top_reg = gather_feat(reg_trans, top_inds, 2);

    auto top_hei = gather_feat(hei, top_inds);

    auto rot_trans = reshape(rot, rot_shape);  
    auto top_rot = gather_feat(rot_trans, top_inds, 2);

    auto dim_trans = reshape(dim, dim_shape);
    auto top_dim = gather_feat(dim_trans, top_inds, 3);

    for (auto i = 0u; i < top_scores.size(); i++) {
      if (top_hei[i] >= post_center_range_[2] &&
          top_hei[i] <= post_center_range_[5]) {
        CenterPointResult tmp_result;
        float xs = (top_xs[i] + top_reg[i * 2]) * out_size_factor_ * voxel_size_[0] + pc_range_[0];
        float ys = (top_ys[i] + top_reg[i * 2 + 1]) * out_size_factor_ * voxel_size_[1] + pc_range_[1];
        //AINFO << xs << " " <<ys;
        if (xs >= post_center_range_[0] && ys >= post_center_range_[1] &&
            xs <= post_center_range_[3] && ys <= post_center_range_[4]) {
          tmp_result.bbox[0] = xs;
          tmp_result.bbox[1] = ys;
          tmp_result.bbox[2] =
              top_hei[i];// - res[res_num].bbox[5] * 0.5f;
          tmp_result.bbox[3] = std::exp(top_dim[3 * i]);
          tmp_result.bbox[4] = std::exp(top_dim[3 * i + 1]);
          tmp_result.bbox[5] = std::exp(top_dim[3 * i + 2]);

          tmp_result.bbox[6] =
              atan2(top_rot[2 * i], top_rot[2 * i + 1]);
          tmp_result.score = top_scores[i];//sigmoid(top_scores[i]);//sigmoid(top_scores[i]);//top_scores[i];
          tmp_result.label = top_clses[i] + first_label;
          res.push_back(tmp_result);
          res_num++;
          // AINFO << tmp_result.bbox[0] << " " << tmp_result.bbox[1] << " " << tmp_result.bbox[2] << " " 
          //       << tmp_result.bbox[3] << " " << tmp_result.bbox[4] << " " << tmp_result.bbox[5] << " " 
          //       << tmp_result.bbox[6] << " " << tmp_result.score << " " <<tmp_result.label;
 
        }
      }
    }
  }
  first_label += cat;
  return res_num;
}

}  // namespace bev
