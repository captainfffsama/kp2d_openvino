/******************************************************************************
* File:             kp2d_test.cpp
*
* Author:           CaptainHu  
* Created:          03/24/21 
* Description:      KP2D test
*****************************************************************************/

#include "KP2D.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <ctime>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

struct Compare{
    bool operator () (int a,int b){
        return a>b;
    }
};



int main(int argc, char *argv[])
{
    
    std::string testImgPath="/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/110kV嘎西I线0472隔离开关C相/20201102_08_00_02_1604275200563/110kV嘎西I线0472隔离开关C相本体.jpg";
    std::string testImg2Path="/home/chiebotgpuhq/MyCode/python/pytorch/KP2D/data/datasets/test/110kV东嘎变电站/110kV嘎西I线0472隔离开关C相/20201031_14_08_31_1604124535476/110kV嘎西I线0472隔离开关C相本体.jpg";
    std::string modelPath="/home/chiebotgpuhq/intel/openvino_2021.2.185/deployment_tools/kp2d_v10.xml";
    cv::Mat src=cv::imread(testImgPath);
    cv::Mat src2=cv::imread(testImg2Path);
    std::cout<<src.size<<std::endl;


    std::clock_t s,e;
    s=clock();
    kp2d::KP2D model(modelPath,20000,0.3);
    e=clock();
    std::cout<<"init "<<(double)(e-s)/CLOCKS_PER_SEC<<std::endl;
    std::vector<cv::KeyPoint> kps1,kps2;
    cv::Mat descs1,descs2;
    std::vector<float> scores1,scores2;
    s=clock();
    bool success=model.Infer(src, kps1,descs1,scores1);
    e=clock();
    std::cout<<"infer "<<(double)(e-s)/CLOCKS_PER_SEC<<std::endl;
    bool success1=model.Infer(src2, kps2,descs2,scores2);

    cv::Ptr<cv::BFMatcher> matcher=cv::BFMatcher::create(cv::NORM_L2,true);

    std::vector<cv::DMatch> matche_kp;
    matcher->match(descs2,descs1,matche_kp);

    std::vector<cv::DMatch> goodMatches;
    for(uint i=0;i<matche_kp.size();++i)
    {
        goodMatches.push_back(matche_kp[i]);
    }

    cv::Mat dstImage;
    cv::drawMatches(src2,kps2,src,kps1,goodMatches,dstImage);

    cv::namedWindow("test",cv::WINDOW_NORMAL);
    cv::imshow("test",dstImage);
    cv::waitKey();

    return 0;
}
