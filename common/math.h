/*
 * @Author: yangcheng 
 * @Date: 2022-09-21 14:37:21 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-13 12:01:40
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

#define M_E        2.71828182845904523536   // e
#define M_LOG2E    1.44269504088896340736   // log2(e)
#define M_LOG10E   0.434294481903251827651  // log10(e)
#define M_LN2      0.693147180559945309417  // ln(2)
#define M_LN10     2.30258509299404568402   // ln(10)
#define M_PI       3.14159265358979323846   // pi
#define M_PI_2     1.57079632679489661923   // pi/2
#define M_PI_4     0.785398163397448309616  // pi/4
#define M_1_PI     0.318309886183790671538  // 1/pi
#define M_2_PI     0.636619772367581343076  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401  // 1/sqrt(2)

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