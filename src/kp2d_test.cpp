/******************************************************************************
* File:             kp2d_test.cpp
*
* Author:           CaptainHu  
* Created:          03/24/21 
* Description:      KP2D test
*****************************************************************************/

#include "kp2d.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <opencv2/core/types.hpp>
#include <vector>

struct Compare{
    bool operator () (int a,int b){
        return a>b;
    }
};


int main(int argc, char *argv[])
{
    
    std::string testImgPath="/home/chiebotgpuhq/pic_tmp/xianlan1.jpeg";
    std::string modelPath="/home/chiebotgpuhq/intel/openvino_2021.2.185/deployment_tools/kp2d_v10.xml";
    cv::Mat src=cv::imread(testImgPath);

    std::cout<<src.total()<<std::endl;

    kp2d::KP2D model(modelPath);
    kp2d::KPResult kpresult;
    bool success=model.Infer(src, kpresult);
    std::cout<<"success flag:"<<success<<std::endl;

    // std::vector<int> a {5,1,4,2,5,3,6,10};
    // std::vector<int> b;
    // std::make_heap(b.begin(),b.end());
    // for (auto c:a){
    //     b.push_back(c);
    //     std::push_heap(b.begin(),b.end(),Compare());
    // }
    // 
    // for (auto c:b){
    //     std::cout<<c<<" ";
    // }
    // std::cout<<std::endl;
    // for (int i=0;i<6;++i)
    // {
    //     std::pop_heap(b.begin(),b.end(),Compare());
    //     std::cout<<b.back()<<std::endl;
    //     b.pop_back();
    // }

    
    return 0;
}
