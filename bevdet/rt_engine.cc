/*
 * @Author: ycdhq 
 * @Date: 2023-06-06 10:33:35 
 * @Last Modified by: ycdhq
 * @Last Modified time: 2023-06-08 10:47:50
 */

#include "rt_engine.h"
#include <fstream>
#include "common/engine_log.h"
#include <algorithm>
static Logger rtengine_gLogger;

namespace bev {

RTEngine::~RTEngine() {
  if (gpu_id_ >= 0) {
    BASE_CUDA_CHECK(cudaStreamDestroy(stream_));
    context_->destroy();
    for (auto buf : buffers_) {
      cudaFree(buf);
    }
  }
}

bool RTEngine::Init(){
  if (gpu_id_ < 0) {
    AINFO << "must use gpu mode";
    return false;
  }
  BASE_CUDA_CHECK(cudaSetDevice(gpu_id_));
  // stream will only be destoried for gpu_id_ >= 0
  cudaStreamCreate(&stream_);    // Load tensorrt engine file

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, gpu_id_);


  AINFO << "engine_file_ " << engine_file_;
  char *trtModelStream = nullptr;
  size_t size = 0 ;
  try{
    std::ifstream file(engine_file_, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    run_time_ = nvinfer1::createInferRuntime(rtengine_gLogger);
    assert(run_time_ != nullptr);
    cuda_engine_ = run_time_->deserializeCudaEngine(trtModelStream, size);
    assert(cuda_engine_ != nullptr);
    context_ = cuda_engine_->createExecutionContext();
    assert(context_ != nullptr);
    delete[] trtModelStream;
  }
  catch (std::exception& e){
    AERROR << "Exception: " << e.what();
    return -1;
  }
  buffers_.resize(input_names_.size() + output_names_.size());
  init_blob(input_names_);
  init_blob(output_names_);
  return true;
}

void RTEngine::init_blob(std::vector<std::string> names) {
  auto engine = &(context_->getEngine());
  for (auto& name : names) {
    AINFO << "name: " << name;
    int bindingIndex =
        engine->getBindingIndex(name.c_str());
    CHECK_LT(static_cast<size_t>(bindingIndex), buffers_.size());
    CHECK_GE(bindingIndex, 0);
    nvinfer1::Dims dims = static_cast<nvinfer1::Dims &&>(
        engine->getBindingDimensions(bindingIndex));

    int count = 1;
    for (size_t i = 0; i < dims.nbDims; i++)
      count = count * dims.d[i];
    // int count = dims.c() * dims.h() * dims.w() * max_batch_size_;
    cudaMalloc(&buffers_[bindingIndex], count * sizeof(float));
    std::vector<int> shape;
    ACHECK(this->shape(name, &shape));
    std::shared_ptr<Blob<float>> blob;
    blob.reset(new Blob<float>(shape));
    blob->set_gpu_data(reinterpret_cast<float *>(buffers_[bindingIndex]));
    blobs_.insert(std::make_pair(name, blob));
  }
}

void RTEngine::Infer() {
  BASE_CUDA_CHECK(cudaSetDevice(gpu_id_));
  BASE_CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (auto name : input_names_) {
    auto blob = get_blob(name);
    if (blob != nullptr) {
      blob->gpu_data();
    }
  }
  // If `out_blob->mutable_cpu_data()` is invoked outside,
  // HEAD will be set to CPU, and `out_blob->mutable_gpu_data()`
  // after `enqueue` will copy data from CPU to GPU,
  // which will overwrite the `inference` results.
  // `out_blob->gpu_data()` will set HEAD to SYNCED,
  // then no copy happends after `enqueue`.
  for (auto name : output_names_) {
    auto blob = get_blob(name);
    if (blob != nullptr) {
      blob->gpu_data();
    }
  }
  context_->enqueue(max_batch_size_, &buffers_[0], stream_, nullptr);
  BASE_CUDA_CHECK(cudaStreamSynchronize(stream_));

  for (auto name : output_names_) {
    auto blob = get_blob(name);
    if (blob != nullptr) {
      blob->mutable_gpu_data();
    }
  }
}

bool RTEngine::shape(const std::string &name, std::vector<int> *res) {
  auto engine = &(context_->getEngine());

  int bindingIndex = engine->getBindingIndex(name.c_str());
  if (bindingIndex > static_cast<int>(buffers_.size())) {
    return false;
  }
  nvinfer1::Dims dims = static_cast<nvinfer1::Dims &&>(
      engine->getBindingDimensions(bindingIndex));
  res->resize(dims.nbDims);
    for (size_t i = 0; i < dims.nbDims; i++)
      (*res)[i] = dims.d[i];  
  return true;
}

std::shared_ptr<Blob<float>> RTEngine::get_blob(
    const std::string &name) {
  auto iter = blobs_.find(name);
  if (iter == blobs_.end()) {
    return nullptr;
  }
  return iter->second;
}

}  // namespace bev