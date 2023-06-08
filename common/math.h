/*
 * @Author: yangcheng 
 * @Date: 2022-09-21 14:37:21 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 10:01:08
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include "common/log.h"

static float Sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + std::exp(-x));
    } else {
        return std::exp(x) / (1.0f + std::exp(x));    /* to aovid overflow */
    }
}

static float Logit(float x) {
    if (x == 0) {
        return static_cast<float>(INT32_MIN);
    } else  if (x == 1) {
        return static_cast<float>(INT32_MAX);
    } else {
        return std::log(x / (1.0f - x));
    }
}


static inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

// 实现softmax,在C维度进行,输入为一维数组，数据按照一个channel一个channel排列
// 实现softmax,在C维度进行,输入为一维数组，数据按照一个channel一个channel排列
static void softmax(float* input_p, int C, int H, int W) {
    for(int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0;
            float max = 0.0;
            for (int k = 0; k < C; k++) {
                if (max < input_p[k * W * H + i * W + j])
                    max = input_p[k * W * H + i * W + j];
            }

            for (int k = 0; k < C; k++) {
                input_p[k * W * H + i * W + j] = exp(input_p[k * W * H + i * W + j] - max);
                sum += input_p[k * W * H + i * W + j];
            }
            for(int k = 0; k < C; k++) {
                input_p[k * W * H + i * W + j] /= sum;  
            }
        }
    }
}

// 实现softmax,在C维度进行,输入为一维数组，数据按照一个channel一个channel排列
static void softmax(float* input_p, int N, int C, int H, int W)
{
    for (int n = 0; n < N; n++) {
        for(int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                float sum = 0.0;
                float max = 0.0;
                for (int k = 0; k < C; k++) {
                    if (max < input_p[n * C * H * W + k * W * H + i * W + j])
                        max = input_p[n * C * H * W + k * W * H + i * W + j];
                }

                for (int k = 0; k < C; k++) {
                    input_p[n * C * H * W + k * W * H + i * W + j] = exp(input_p[n * C * H * W + k * W * H + i * W + j] - max);
                    sum += input_p[n * C * H * W + k * W * H + i * W + j];
                }
                for(int k = 0; k < C; k++) {
                    input_p[n * C * H * W + k * W * H + i * W + j] /= sum;  
                }
            }
        }
    }
}



static void tran_pose(float* input, float* out, int N, int C, int H, int W){

    
  for(int n = 0; n < N; n++){
    for(int c = 0; c < C; c++) {
      for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
          int src_index = n * C * H * W + c * H * W + h * W + w;
          int dst_index = n * H * W * C + h * W * C + w * C + c;
          out[dst_index] = input[src_index];
        }
      }
    }
  }

    // for(int n = 0; n < N; n++){
    //     for(int h = 0; h < H; h++) {
    //     for(int w = 0; w < W; w++) {
    //         for(int c = 0; c < C; c++) {
    //         int dst_index = n * C * H * W + c * H * W + h * W + w;
    //         int src_index = n * H * W * C + h * W * C + w * C + c;
    //         out[dst_index] = input[src_index];
    //         }
    //     }
    //     }
    // }
}

static float* SoftMaxFast(float* src, int32_t length) {
    float alpha = *std::max_element(src, src + length);
    AINFO << alpha;
    float max = -10000;
    for(int i = 0; i < length;i++) {
        if (src[i] > max) 
            max = src[i];
    }
    AINFO <<max;
    float denominator = 0;

    float* dst;
    dst = new float[length];
    for (int32_t i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int32_t i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }
    return dst;
}


static float diou(float lbox[4], float rbox[4]) {
   float interBox[] = {
        (std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
        (std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
        (std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
        (std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}