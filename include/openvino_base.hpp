/******************************************************************************
* File:             openvino_base.hpp
*
* Author:           CaptainHu  
* Created:          03/23/21 
* Description:      openvino基类
*****************************************************************************/

#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "inference_engine.hpp"

namespace openvino {
    using namespace InferenceEngine;
    class VinoBase
    {
    private:
        Core ie;
        CNNNetwork network;
        ExecutableNetwork exeNetwork;

    protected:
        InputsDataMap inputMap;
        OutputsDataMap outputMap;
        InferRequest inferRequest;
    
    public:
        explicit VinoBase(const std::string& xmlPath);
        virtual ~VinoBase();


        virtual void reshapeNet(int batchsize,int rows,int cols) final;
        virtual void setInputInfo();
        virtual void setOutputInfo();

        virtual void infer(const cv::Mat src);

    };
}
