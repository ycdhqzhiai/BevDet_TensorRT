/*
 * @Author: ycdhq 
 * @Date: 2023-06-08 14:02:57 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-14 16:31:20
 */
#pragma once
#include <iostream>
#include <cmath>
#include <algorithm>
#include "nuscenes_box.h"
constexpr float EPS = 1e-8;

struct Point {
  float x, y;
  Point() {}
  Point(double _x, double _y) { x = _x, y = _y; }

  void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  Point operator+(const Point& b) const { return Point(x + b.x, y + b.y); }

  Point operator-(const Point& b) const { return Point(x - b.x, y - b.y); }
};




template<typename T>
inline float sigmoid(T x) { 
    return (1.0 / (1 + exp(-static_cast<float>(x)))); 
}

// do element-wise sigmoid on tensor
template<typename T>
void sigmoid_on_tensor(T* input, T* out, int64_t tensor_size) {
    for(int64_t i = 0; i < tensor_size; i++) {
        out[i] = sigmoid<T>(input[i]);
    }
}

void xywhr2xyxyr(const float box_xywhr[7], float* dst, float factor) {
  //AINFO << box_xywhr[0] << " " << box_xywhr[1]  << " " << box_xywhr[2] << " " << box_xywhr[3] << " " << box_xywhr[4];
  auto half_w = box_xywhr[3] * factor / 2;
  auto half_h = box_xywhr[4] * factor / 2;
  dst[0] = box_xywhr[0] - half_w;
  dst[1] = box_xywhr[1] - half_h;
  dst[2] = box_xywhr[0] + half_w;
  dst[3] = box_xywhr[1] + half_h;
  dst[4] = box_xywhr[6];
  //AINFO << dst[0] << " " << dst[1]  << " " << dst[2] << " " << dst[3] << " " << dst[4];
}


static int check_in_box2d(const float for_box2d[], const float box[],
                          const Point& p) {
  // params: box (5) [x1, y1, x2, y2, angle]
  const float MARGIN = 1e-5;

  auto& center_x = for_box2d[0];
  auto& center_y = for_box2d[1];
  auto& angle_cos = for_box2d[2];
  auto& angle_sin = for_box2d[3];
  // rotate the point in the opposite direction of box
  float rot_x = p.x * angle_cos + p.y * angle_sin + center_x;
  float rot_y = -p.x * angle_sin + p.y * angle_cos + center_y;
  return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
          rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

inline float cross(const Point& a, const Point& b) {
  return a.x * b.y - a.y * b.x;
}

inline float cross(const Point& p1, const Point& p2, const Point& p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int point_cmp(const Point& a, const Point& b, const Point& center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

static int check_rect_cross(const Point& p1, const Point& p2, const Point& q1,
                            const Point& q2) {
  int ret = std::min(p1.x, p2.x) <= std::max(q1.x, q2.x) &&
            std::min(q1.x, q2.x) <= std::max(p1.x, p2.x) &&
            std::min(p1.y, p2.y) <= std::max(q1.y, q2.y) &&
            std::min(q1.y, q2.y) <= std::max(p1.y, p2.y);
  return ret;
}

static int intersection(const Point& p1, const Point& p0, const Point& q1,
                        const Point& q0, Point& ans) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans.x = (b0 * c1 - b1 * c0) / D;
    ans.y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

inline void calc_first(float box[], Point box_corners[], float for_box2d[]) {
  float x1 = box[0];
  float y1 = box[1];
  float x2 = box[2];
  float y2 = box[3];
  float angle = box[4];
  auto center = Point((x1 + x2) / 2, (y1 + y2) / 2);
  box_corners[0].set(x1, y1);
  box_corners[1].set(x2, y1);
  box_corners[2].set(x2, y2);
  box_corners[3].set(x1, y2);
  float angle_cos = cos(angle), angle_sin = sin(angle);
  // get oriented corners
  for (int k = 0; k < 4; k++) {
    float new_x = (box_corners[k].x - center.x) * angle_cos +
                  (box_corners[k].y - center.y) * angle_sin + center.x;
    float new_y = -(box_corners[k].x - center.x) * angle_sin +
                  (box_corners[k].y - center.y) * angle_cos + center.y;
    box_corners[k].set(new_x, new_y);
  }

  box_corners[4] = box_corners[0];
  for_box2d[2] = cos(-angle);
  for_box2d[3] = sin(-angle);
  for_box2d[0] = -(box[0] + box[2]) / 2. * for_box2d[2] -
                 (box[1] + box[3]) / 2. * for_box2d[3] + (box[0] + box[2]) / 2.;
  for_box2d[1] = (box[0] + box[2]) / 2. * for_box2d[3] -
                 (box[1] + box[3]) / 2. * for_box2d[2] + (box[1] + box[3]) / 2.;
}

static float box_overlap(const std::vector<bool>& is_first, float boxes[][5],
                         size_t a, size_t b, Point box_corners[][5],
                         float for_box2d[][4]) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]

  if (is_first[a]) calc_first(boxes[a], box_corners[a], for_box2d[a]);
  if (is_first[b]) calc_first(boxes[b], box_corners[b], for_box2d[b]);

  auto& box_a_corners = box_corners[a];
  auto& box_b_corners = box_corners[b];

  // get intersection of lines
  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(for_box2d[a], boxes[a], box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(for_box2d[b], boxes[b], box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }
  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

float iou_bev(std::vector<bool>& is_first, float box_areas[],
                     Point box_corners[][5], float for_box2d[][4],
                     float boxes[][5], size_t a, size_t b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]
  if (is_first[a])
    box_areas[a] = (boxes[a][2] - boxes[a][0]) * (boxes[a][3] - boxes[a][1]);
  if (is_first[b])
    box_areas[b] = (boxes[b][2] - boxes[b][0]) * (boxes[b][3] - boxes[b][1]);
  float& sa = box_areas[a];
  float& sb = box_areas[b];
  float s_overlap = box_overlap(is_first, boxes, a, b, box_corners, for_box2d);
  auto res = s_overlap / fmaxf(sa + sb - s_overlap, EPS);
  if (is_first[a]) is_first[a] = false;
  if (is_first[b]) is_first[b] = false;
  return res;
}

static std::vector<uint32_t> division_n(const std::vector<uint32_t>& src,
                                        uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i] / dividend;
  }
  return dst;
}


static std::vector<uint32_t> remainder_n(const std::vector<uint32_t>& src,
                                         uint32_t dividend) {
  std::vector<uint32_t> dst(src.size());
  for (size_t i = 0; i < src.size(); i++) {
    dst[i] = src[i] % dividend;
  }
  return dst;
}

template <typename T1>
std::vector<T1> gather_feat(const T1 input[],
                            const std::vector<uint32_t>& index, size_t cat) {
  std::vector<T1> dst(index.size() * cat);
  for (auto i = 0u; i < index.size(); i++) {
    for (size_t j = 0; j < cat; j++) {
      dst[cat * i + j] = input[cat * index[i] + j];
    }
  }
  return dst;
}
template <typename T1>
std::vector<T1> gather_feat(const T1 input[],
                            const std::vector<uint32_t>& index) {
  return gather_feat(input, index, 1);
}
template <typename T1>
std::vector<T1> gather_feat(const std::vector<T1>& input,
                            const std::vector<uint32_t>& index, size_t cat) {
  std::vector<T1> dst(index.size() * cat);
  for (auto i = 0u; i < index.size(); i++) {
    for (size_t j = 0; j < cat; j++) {
      dst[cat * i + j] = input[cat * index[i] + j];
    }
  }
  return dst;
}
template <typename T1>
std::vector<T1> gather_feat(const std::vector<T1>& input,
                            const std::vector<uint32_t>& index) {
  return gather_feat(input, index, 1);
}


std::pair<std::vector<float>, std::vector<uint32_t>> topK(
    const float* scores, int scores_num, int begin, uint32_t topk, float score_threshold) {
  using elem_t = std::pair<float, uint32_t>;
  std::vector<elem_t> queue;
  for (auto j = scores_num * begin; j < scores_num * (begin + 1); j++) {
    if (scores[j] > score_threshold)
      queue.push_back({scores[j], j});    
  }
  auto min_size = topk > queue.size() ? queue.size() : topk;
  std::stable_sort(queue.begin(), queue.end(),
                   [](const elem_t& x, const elem_t& y) -> bool {
                     return x.first > y.first;
                   });

  std::vector<float> scalars(min_size);
  std::vector<uint32_t> indices(min_size);
  for (auto j = 0u; j < min_size; ++j) {
    scalars[j] = queue[j].first;
    indices[j] = queue[j].second % scores_num;
  }
  return std::make_pair(scalars, indices);
}

std::pair<std::vector<float>, std::vector<uint32_t>> topK(
    const float* scores, int scores_num, uint32_t topk, float score_threshold) {
  using elem_t = std::pair<float, uint32_t>;
  std::vector<elem_t> queue;
  for (auto j = 0u; j < scores_num; ++j) {
    if (scores[j] > score_threshold)
      queue.push_back({scores[j], j});
  }
  auto min_size = topk > queue.size() ? queue.size() : topk;
  std::stable_sort(queue.begin(), queue.end(),
                   [](const elem_t& x, const elem_t& y) -> bool {
                     return x.first > y.first;
                   });

  std::vector<float> scalars(min_size);
  std::vector<uint32_t> indices(min_size);
  for (auto j = 0u; j < min_size; ++j) {
    scalars[j] = queue[j].first;
    indices[j] = queue[j].second;
  }
  return std::make_pair(scalars, indices);
}


std::pair<std::vector<float>, std::vector<uint32_t>> topK(
    const std::vector<float>& scores, uint32_t topk, float score_threshold) {
  using elem_t = std::pair<float, uint32_t>;
  std::vector<elem_t> queue;
  for (auto j = 0u; j < scores.size(); ++j) {
    if (scores[j] > score_threshold)
      queue.push_back({scores[j], j});
  }
  auto min_size = topk > queue.size() ? queue.size() : topk;
  std::stable_sort(queue.begin(), queue.end(),
                   [](const elem_t& x, const elem_t& y) -> bool {
                     return x.first > y.first;
                   });

  std::vector<float> scalars(min_size);
  std::vector<uint32_t> indices(min_size);
  for (auto j = 0u; j < min_size; ++j) {
    scalars[j] = queue[j].first;
    indices[j] = queue[j].second;
  }
  return std::make_pair(scalars, indices);
}


template <typename T1>
std::vector<T1> reshape(const T1* input,
                          int shape) {

  int B = 1;
  int C = shape;
  int W = 128;
  int H = 128;                        
  std::vector<T1> out;
  out.resize(C *128*128);

  for(int b = 0; b < B; b++){
    for(int c = 0; c < C; c++) {
      for(int w = 0; w < W; w++) {
        for(int h = 0; h < H; h++) {
          int src_index = b * C * H * W + c * H * W + h * W + w;
          int dst_index = b * H * W * C + h * W * C + w * C + c;
          out[dst_index] = input[src_index];
        }
      }
    }
  }
  return out;
}

inline void FindMaxMin(float (&box)[4][2], float& maxVAl, float& minVAl, int xyIdx){
    
    maxVAl = box[0][xyIdx];
    minVAl = box[0][xyIdx];
    
    for(auto idx=0; idx < 4; idx++){
        if (maxVAl < box[idx][xyIdx])
            maxVAl = box[idx][xyIdx];

        if (minVAl > box[idx][xyIdx])
            minVAl = box[idx][xyIdx];
    }
}


inline void AlignBox(float (&cornerRot)[4][2], float (&cornerAlign)[2][2]){

    float maxX = 0;
    float minX = 0;
    float maxY = 0;
    float minY = 0;

    FindMaxMin(cornerRot, maxX, minX, 0); // 0 mean X
    FindMaxMin(cornerRot, maxY, minY, 1); // 1 mean X

    cornerAlign[0][0] = minX;
    cornerAlign[0][1] = minY;
    cornerAlign[1][0] = maxX;
    cornerAlign[1][1] = maxY;
}


void RotateAroundCenter(bev::BBox& box, float (&corner)[4][2], float& cosVal, float& sinVal, float (&cornerANew)[4][2]){
    
    for(auto idx = 0; idx < 4; idx++){
        auto x = corner[idx][0];
        auto y = corner[idx][1];

        cornerANew[idx][0] = (x - box.bbox[0]) * cosVal + (y - box.bbox[1]) * (-sinVal) + box.bbox[0];
        cornerANew[idx][1] = (x - box.bbox[0]) * sinVal + (y - box.bbox[1]) * cosVal + box.bbox[1];
    }
}


float IoUBev(bev::BBox& boxA, bev::BBox& boxB){
   
    float ax1 = boxA.bbox[0] - boxA.bbox[3]/2;
    float ax2 = boxA.bbox[0] + boxA.bbox[3]/2;
    float ay1 = boxA.bbox[1] - boxA.bbox[4]/2;
    float ay2 = boxA.bbox[1] + boxA.bbox[4]/2;

    float bx1 = boxB.bbox[0] - boxB.bbox[3]/2;
    float bx2 = boxB.bbox[0] + boxB.bbox[3]/2;
    float by1 = boxB.bbox[1] - boxB.bbox[4]/2;
    float by2 = boxB.bbox[1] + boxB.bbox[4]/2;

    float cornerA[4][2] = {{ax1, ay1}, {ax1, ay2},
                         {ax2, ay1}, {ax2, ay2}};
    float cornerB[4][2] = {{bx1, ay1}, {bx1, by2},
                         {bx2, by1}, {bx2, by2}};
    
    float cornerARot[4][2] = {0};
    float cornerBRot[4][2] = {0};

    float cosA = cos(boxA.bbox[6]), sinA = sin(boxA.bbox[6]);
    float cosB = cos(boxB.bbox[6]), sinB = sin(boxB.bbox[6]);

    RotateAroundCenter(boxA, cornerA, cosA, sinA, cornerARot);
    RotateAroundCenter(boxB, cornerB, cosB, sinB, cornerBRot);

    float cornerAlignA[2][2] = {0};
    float cornerAlignB[2][2] = {0};

    AlignBox(cornerARot, cornerAlignA);
    AlignBox(cornerBRot, cornerAlignB);
    
    float sBoxA = (cornerAlignA[1][0] - cornerAlignA[0][0]) * (cornerAlignA[1][1] - cornerAlignA[0][1]);
    float sBoxB = (cornerAlignB[1][0] - cornerAlignB[0][0]) * (cornerAlignB[1][1] - cornerAlignB[0][1]);
    
    float interW = std::min(cornerAlignA[1][0], cornerAlignB[1][0]) - std::max(cornerAlignA[0][0], cornerAlignB[0][0]);
    float interH = std::min(cornerAlignA[1][1], cornerAlignB[1][1]) - std::max(cornerAlignA[0][1], cornerAlignB[0][1]);
    
    float sInter = std::max(interW, 0.0f) * std::max(interH, 0.0f);
    float sUnion = sBoxA + sBoxB - sInter;
    
    return sInter/sUnion;
}



void AlignedNMSBev(std::vector<bev::BBox>& predBoxs){
    
    if(predBoxs.size() == 0)
        return;

    std::sort(predBoxs.begin(),predBoxs.end(),[ ](bev::BBox& box1, bev::BBox& box2){return box1.score > box2.score;});

    auto boxSize = predBoxs.size();
    int numBoxValid = 0;
    for(auto boxIdx1 =0; boxIdx1 < boxSize; boxIdx1++){
        
        if (predBoxs[boxIdx1].isDrop) continue;

        for(auto boxIdx2 = boxIdx1+1; boxIdx2 < boxSize; boxIdx2++){
            if(predBoxs[boxIdx2].isDrop == true)
                continue;
            float iou = IoUBev(predBoxs[boxIdx1], predBoxs[boxIdx2]);
            if(iou > 0.25f)
                predBoxs[boxIdx2].isDrop = true;
        } 
        if (!predBoxs[boxIdx1].isDrop) numBoxValid ++;
    }
}