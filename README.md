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
+ nms存在一定问题，对于小目标(行人)过滤不理想
+ python原仓库显示部分太复杂，涉及到的坐标系太多，eog->global->lidar->camera->image,不知道显示图像的token，相机内参不准确，显示有一定问题