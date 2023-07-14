# BEVDet TensorRT实现
## 1.流程
将bevdet分为四个部分，一第一个部分resnet50提取2D图像特征，第二部分bev_poolv2实现2D到3D转换，第三部分残差模块Encode，第四部分后处理

其中第一第三部分使用Tensorrt实现，其他在cpu上跑

## 2.run

```shell
bash convert.sh
mkdir build && cd build
cmake ..
make -j12
./bevdet $you_modelpath
```

## 3.result
![](./data/show_img)

## 4.问题
+ 后处理实现参考https://github.com/ycdhqzhiai/BEVDet_PostProcess
