/******************************************************************************
* File:             kp2d_test.cpp
*
* Author:           CaptainHu  
* Created:          03/24/21 
* Description:      KP2D test
*****************************************************************************/

#include "kp2d.hpp"
#include "opencv2/opencv.hpp"

int main(int argc, char *argv[])
{
    
    std::string testImgPath="/home/chiebotgpuhq/pic_tmp/xianlan.jpeg";
    std::string modelPath="/home/chiebotgpuhq/intel/openvino_2021.2.185/deployment_tools/kp2d.onnx";
    cv::Mat src=cv::imread(testImgPath);

    kp2d::KP2D model(modelPath);
    kp2d::KPResult result;
    model.infer(src, result);
    return 0;
}
