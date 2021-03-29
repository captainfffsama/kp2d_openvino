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

        int cellSize; // 每个cell的尺寸,数值是2的下采样数次方
        cv::Size* inputImgSize;
        int topK; //  选分数最大的前多少个点
        float threshold; // 选分数时候的阈值
        int scoreDim; // 分数张量的轴数
        int featDim;  // 描述子张量的轴数
        int coordDim; // 偏移张量的轴数
        int batchSize;
        float cellStep;  // (cellSize-1)/2
        float crossRatio;
        int featCellSize;

        Blob::Ptr Mat2Blob(const cv::Mat& mat);
        bool Blob2Mat(const Blob::Ptr& blob,cv::Mat& blobMat);
        void SetInputInfo();
        void SetOutputInfo();

        Blob::Ptr PreProcess(const cv::Mat& src);
        bool PostProcess(const Blob::Ptr& coordsBlob,const Blob::Ptr& scoreBlob,const Blob::Ptr& featBlob,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores);

        inline float BilinearInter(float x,float y,float x1,float y1,float x2,float y2,float f11,float f21,float f12,float f22);

        template<typename T>
        const T* GetBlobReaderPtr(const Blob::Ptr& blob);
        template<typename T>
        T* GetBlobWritePtr(Blob::Ptr& blob);

    public:
        explicit KP2D(const std::string& xmlPath,int top_k=20000,float scoreThr=0.6,int downSample=8,int featLength=256,int batch =1,float crossRate=2);
        ~KP2D();

        bool Infer(const cv::Mat& src,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores);

    };

}
