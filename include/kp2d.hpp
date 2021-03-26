/******************************************************************************
* File:             openvino_base.hpp
*
* Author:           CaptainHu  
* Created:          03/23/21 
* Description:      openvino基类
*****************************************************************************/

#pragma once
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include "inference_engine.hpp"

namespace kp2d {
    using namespace InferenceEngine;
    using KPResult=std::tuple<cv::KeyPoint,cv::Mat,std::vector<float> >;
    using ScoreInfo=std::tuple<int,int,float>;
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
        cv::Size* inputImgSize;
        int topK;
        float threshold;
        int scoreDim;
        int featDim;
        int coordDim;
        int batchSize;

        Blob::Ptr Mat2Blob(const cv::Mat& mat);
        bool Blob2Mat(const Blob::Ptr& blob,cv::Mat& blobMat);
        void SetInputInfo();
        void SetOutputInfo();

        Blob::Ptr PreProcess(const cv::Mat& src);
        bool PostProcess(const Blob::Ptr& coordBlob,const Blob::Ptr& scoreBlob,const Blob::Ptr& featBlob,KPResult& result);

        template<typename T>
        const T* GetBlobReaderPtr(const Blob::Ptr& blob);
        template<typename T>
        T* GetBlobWritePtr(Blob::Ptr& blob);

    public:
        explicit KP2D(const std::string& xmlPath,int top_k=20000,float scoreThr=0.6,int downSample=8,int featLength=256,int batch =1);
        ~KP2D();

        bool Infer(const cv::Mat& src,KPResult& kpresult);

    };

}
