/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 10:27:42 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-06 15:30:10
 */
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "base/blob.h"
namespace bev {
typedef std::map<std::string, std::shared_ptr<Blob<float>>> BlobMap;

class RTEngine {
 public:
    RTEngine() = default;
    RTEngine(const RTEngine&) = delete;
    RTEngine &operator=(const RTEngine&) = delete;
  virtual ~RTEngine();

  bool Init();

  void Infer();

  std::shared_ptr<Blob<float>> get_blob(
      const std::string &name);

 protected:
  bool shape(const std::string &name, std::vector<int> *res);
  void init_blob(std::vector<std::string> names);

  std::string engine_file_;
  std::vector<std::string> output_names_;
  std::vector<std::string> input_names_;

 private:

  nvinfer1::IExecutionContext *context_ = nullptr;
  nvinfer1::ICudaEngine *cuda_engine_;
  nvinfer1::IRuntime *run_time_;


  cudaStream_t stream_ = 0;

  std::vector<std::string> tensor_map_;
  int input_index_;
  std::vector<int> output_index_;

  int max_batch_size_ = 1;
  int gpu_id_ = 0;

//   std::shared_ptr<NetParameter> net_param_;
  std::vector<void *> buffers_;
  int workspaceSize_ = 1;
  BlobMap blobs_;
};

}  // bev
