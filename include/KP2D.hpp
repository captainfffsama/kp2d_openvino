/******************************************************************************
* File:             openvino_base.hpp
*
* Author:           CaptainHu  
* Created:          03/23/21 
* Description:      openvino基类
*****************************************************************************/

#pragma once
#ifndef __KP2D_H
#define __KP2D_H
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include "inference_engine.hpp"

namespace kp2d {
    using ScoreInfo=std::tuple<int,int,float>;
    class KP2D
    {
    private:
        InferenceEngine::Core ie;
        InferenceEngine::CNNNetwork network;
        InferenceEngine::ExecutableNetwork exeNetwork;

        InferenceEngine::InputsDataMap inputMap;
        InferenceEngine::OutputsDataMap outputMap;
        InferenceEngine::InferRequest inferRequest;

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

        InferenceEngine::Blob::Ptr Mat2Blob(const cv::Mat& mat);
        bool Blob2Mat(const InferenceEngine::Blob::Ptr& blob,cv::Mat& blobMat);
        void SetInputInfo();
        void SetOutputInfo();

        InferenceEngine::Blob::Ptr PreProcess(const cv::Mat& src);
        bool PostProcess(const InferenceEngine::Blob::Ptr& coordsBlob,const InferenceEngine::Blob::Ptr& scoreBlob,const InferenceEngine::Blob::Ptr& featBlob,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores);

        inline float BilinearInter(float x,float y,float x1,float y1,float x2,float y2,float f11,float f21,float f12,float f22);

        template<typename T>
        const T* GetBlobReaderPtr(const InferenceEngine::Blob::Ptr& blob);
        template<typename T>
        T* GetBlobWritePtr(InferenceEngine::Blob::Ptr& blob);

    public:
        /*!
         * @brief KP2D 构造函数
         *
         * @param xmlPath [openvino模型xml的路径]
         * @param top_k [最多选多少个点,默认20000]
         * @param scoreThr [分数阈值,默认0.6]
         * @param downSample [模型下采样之后最小特征图对应的像素数,默认8]
         * @param featLength [描述子的长度,默认是256,模型改了要改]
         * @param batch [推理时的batchsize,现在这个没用,就是1]
         * @param crossRate [模型回归的时候步长是cell的几倍,和模型相关,默认是2]
         */
        explicit KP2D(const std::string& xmlPath,int top_k=20000,float scoreThr=0.6,int downSample=8,int featLength=256,int batch =1,float crossRate=2);
        ~KP2D();

        /*!
         * @brief 推理函数
         *
         * @param src [输入图片,大小是固定的,默认1920*1080]
         * @param kps [输出: std::vector<cv::KeyPoint> 关键点]
         * @param descs [输出: cv::Mat M*N大小,M是点的个数,N是描述子长度,一般是256]
         * @param scores [输出: std::vector<float> 每个点的置信度]
         * @return [bool, 推理成功是true,推理失败是false]
         */
        bool Infer(const cv::Mat& src,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores);

    };

}
#endif
