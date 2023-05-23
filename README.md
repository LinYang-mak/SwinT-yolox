# SwinT-yolox
====
Swin Transformer、YOLOX
----
自动扶梯在公共场合被广泛使用，乘客摔倒事故如果不能及时发现并处理，会造成严重的人身伤害，因此实现自动扶梯智能化监控管理势在必行。融合Swin Transformer和YoloX目标检测算法的优秀策略，提出了一种基于SwinT-YoloX网络模型的自动扶梯行人摔倒检测算法。

相关仓库
----

 | 模型 | 路径 |
 | --- | --- |
 | YoloV3 | https://github.com/bubbliiiing/yolo3-pytorch |
 | Efficientnet-Yolo3 | https://github.com/bubbliiiing/efficientnet-yolo3-pytorch |
 | YoloV4	| https://github.com/bubbliiiing/yolov4-pytorch |
 | YoloV4-tiny | https://github.com/bubbliiiing/yolov4-tiny-pytorch |
 | Mobilenet-Yolov4 | https://github.com/bubbliiiing/mobilenet-yolov4-pytorch |
 | YoloV5-V5.0 | https://github.com/bubbliiiing/yolov5-pytorch |
 | YoloV5-V6.1 | https://github.com/bubbliiiing/yolov5-v6.1-pytorch |
 | YoloX | https://github.com/bubbliiiing/yolox-pytorch |
 | YoloV7 | https://github.com/bubbliiiing/yolov7-pytorch |
 | YoloV7-tiny | https://github.com/bubbliiiing/yolov7-tiny-pytorch |
 | Yolov8 | https://github.com/ultralytics/ultralytics |

性能情况
----
 
 | 输入图片大小 | mAP0.5:0.95 | mAP 0.5 |
 | --- | --- | --- |
 | 640X640 | 74.83 | 96.71 |
 
所需环境
---
torch == 1.12.0
其他配置请查看requirements.txt。

文件下载
----
预训练权重下载
针对：phi = "s"
链接：https://pan.baidu.com/s/1h59nARjjdcS-vXZ7tMr3jg 
提取码：mdv6

扶梯摔倒检测所用模型
链接：https://pan.baidu.com/s/1lhin605BSkgU3-o3Py1Dpg 
提取码：varr
