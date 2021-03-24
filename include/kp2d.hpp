/******************************************************************************
* File:             openvino_base.hpp
*
* Author:           CaptainHu  
* Created:          03/23/21 
* Description:      openvino基类
*****************************************************************************/

#pragma once
#include <opencv2/core/mat.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <tuple>
#include "inference_engine.hpp"

namespace kp2d {
    using namespace InferenceEngine;
    using KPResult=std::tuple<cv::KeyPoint,cv::Mat,cv::KeyPoint,cv::Mat>;
    class KP2D
    {
    private:
        Core ie;
        CNNNetwork network;
        ExecutableNetwork exeNetwork;

        InputsDataMap inputMap;
        OutputsDataMap outputMap;
        InferRequest inferRequest;
        int cellSize;

        Blob::Ptr warpMat2Blob(const cv::Mat& mat);
        void setInputInfo();
        void setOutputInfo();

        void preProcess(const cv::Mat& src,Blob::Ptr& srcBlob,cv::Size& newSize);

    public:
        explicit KP2D(const std::string& xmlPath,int downSample=8);
        ~KP2D();

        void infer(const cv::Mat& src,KPResult& kpresult);

    };
}
