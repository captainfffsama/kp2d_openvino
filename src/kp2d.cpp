#include "kp2d.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <tuple>

using namespace kp2d;
using namespace InferenceEngine;

KP2D::KP2D(const std::string& xmlPath,int top_k,float scoreThr,int downSample,int featLength,int batch,float crossRate)
    :cellSize(downSample),topK(top_k),threshold(scoreThr),scoreDim(1),featDim(featLength),coordDim(2),batchSize(batch),cellStep((downSample-1)/2.),crossRatio(crossRate),featCellSize(4)
{
    network=ie.ReadNetwork(xmlPath);
    std::string inputName;
    SizeVector inputShape;
    std::tie(inputName,inputShape)=*network.getInputShapes().begin();
    inputImgSize=new cv::Size(inputShape[3],inputShape[2]);
    SetInputInfo();
    SetOutputInfo();
    exeNetwork=ie.LoadNetwork(network,"CPU");
    inferRequest=exeNetwork.CreateInferRequest();

}

KP2D::~KP2D(){
    delete inputImgSize;
}

void KP2D::SetInputInfo(){
    for(auto& ele:network.getInputsInfo()){
        auto& inputData=ele.second;
        inputData->setPrecision(Precision::FP32);
        inputData->setLayout(Layout::NCHW);
        inputData->getPreProcess().setColorFormat(ColorFormat::BGR);
    }
}

void KP2D::SetOutputInfo(){
    for(auto& ele:network.getOutputsInfo()){
        auto& outData=ele.second;
        outData->setPrecision(Precision::FP32);
        outData->setLayout(Layout::NCHW);
    }
}

bool KP2D::Infer(const cv::Mat& src,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores)
{
    cv::Mat dst;
    src.convertTo(dst, CV_32FC3,1.0/255.);
    //  根据图片尺寸调整网络
    //  TODO: 后续可以考虑加个dynamic参数,对于固定的图片就固定住网络

    Blob::Ptr inputBlob=inferRequest.GetBlob("x");
    auto buffer=inputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
    
    cv::Size inputSize(1920,1080);
    for (int h=0;h<src.rows;++h)
    {
        auto hdataPtr=src.ptr<uchar>(h);
        for(int w=0;w<src.cols;++w)
        {
            for(int c=0;c<src.channels();++c)
            {
                buffer[c*src.rows*src.cols+h*src.cols+w]=static_cast<float>(hdataPtr[w*src.channels()+c]/255.);
            }
        }
    }


    std::clock_t s,e;
    s=clock();
    inferRequest.Infer();
    e=clock();
    std::cout<<"infer:"<<(double)(e-s)/CLOCKS_PER_SEC<<std::endl;
    Blob::Ptr coordBlob=inferRequest.GetBlob("coord");
    Blob::Ptr scoreBlob=inferRequest.GetBlob("score");
    Blob::Ptr featBlob=inferRequest.GetBlob("feat");

    std::cout<<"score blob precision:"<<scoreBlob->getTensorDesc().getPrecision()<<std::endl;

    s=clock();
    if(!PostProcess(coordBlob,scoreBlob,featBlob,kps,descs,scores))
    {
        std::cout<<"ERROR!! post process fail!!"<<std::endl;
        return false;
    }
    e=clock();
    std::cout<<"postProcess:"<<(double)(e-s)/CLOCKS_PER_SEC<<std::endl;
    return true;

}

Blob::Ptr KP2D::PreProcess(const cv::Mat& src){
    if (*inputImgSize!=src.size())
    {
        std::cout<<"ERROR! Image size must be:"<<*inputImgSize<<std::endl;
        return nullptr;
    }
    cv::Mat dst;
    src.convertTo(dst,CV_32F);
    
    // cv::normalize(dst,dst,1,0,cv::NORM_INF);
    Blob::Ptr blob = Mat2Blob(dst);

    InferenceEngine::SizeVector blobSize = blob->getTensorDesc().getDims();
    const size_t width = blobSize[3];
    const size_t height = blobSize[2];
    const size_t channels = blobSize[1];
    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob) {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->rwmap();

    float* blob_data = mblobHolder.as<float *>();

    for (size_t c = 0; c < channels; c++) {
        for (size_t  h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                blob_data[c * width * height + h * width + w] /=255.;
            }
        }
    }
    return blob;

}


Blob::Ptr KP2D::Mat2Blob(const cv::Mat& mat){
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;
    size_t strideH = mat.step.buf[0];
    size_t strideW = mat.step.buf[1];

    // bool is_dense =
    //         strideW == channels &&
    //         strideH == channels * width;
    // if (!is_dense) THROW_IE_EXCEPTION
    //             << "Doesn't support conversion from not dense cv::Mat";

    TensorDesc tDesc(Precision::FP32,
                    {1, channels, height, width},
                    Layout::NCHW);

    return make_shared_blob<float>(tDesc,(float*)mat.data);
}

bool KP2D::PostProcess(const Blob::Ptr& coordsBlob,const Blob::Ptr& scoreBlob,const Blob::Ptr& featBlob,std::vector<cv::KeyPoint>& kps,cv::Mat& descs,std::vector<float>& scores)
{
    const float* scorePtr=GetBlobReaderPtr<float>(scoreBlob);
    const float* featPtr=GetBlobReaderPtr<float>(featBlob);
    const float* coordsPtr=GetBlobReaderPtr<float>(coordsBlob);

    if (!((scorePtr && featPtr) && coordsPtr))
    {
        std::cout<<"ERROR!!! get data failed"<<std::endl;
        return false;
    }

    auto scoreSize=scoreBlob->getTensorDesc().getDims();

    const int sw=scoreSize[3];
    const int sh=scoreSize[2];


    const int imgW=sw*cellSize;
    const int imgH=sh*cellSize;

    std::vector<ScoreInfo> scoreIdxTemp_v;
    std::make_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end());

    int kpNumberCount=topK;
    int borderDrop=std::ceil(crossRatio-1); // 简单处理,直接把可能回归超出图像的点去掉
    for (int h=borderDrop;h<(sh-borderDrop);h++){
        for (int w=borderDrop;w<(sw-borderDrop);w++){
            if(scorePtr[h*sw+w]>threshold){
                scoreIdxTemp_v.push_back(std::make_tuple(w,h,scorePtr[h*sw+w]));
                std::push_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end(),[](const ScoreInfo& A,const ScoreInfo& B) {return std::get<2>(A) > std::get<2>(B); });
                kpNumberCount--;
            }
        }
    }

    //  若过阈值的数少于topK  需要选出最大的k个
    while(kpNumberCount<0)
    {
        std::pop_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end(),
                [](const ScoreInfo& A,const ScoreInfo& B) {return std::get<2>(A) > std::get<2>(B); });
        scoreIdxTemp_v.pop_back();
        kpNumberCount++;
    }

    int idxx=0,idxy=0,score=0;
    auto coordSize=coordsBlob->getTensorDesc().getDims();
    const int &cw=coordSize[3], & ch=coordSize[2];
    auto featSize=featBlob->getTensorDesc().getDims();
    const int &fw=featSize[3], & fh=featSize[2],& fc=featSize[1];
    cv::Mat descriptors(scoreIdxTemp_v.size(),fc,CV_32FC1);

    int currentKpIdx=0;
    for(auto scoreInfo:scoreIdxTemp_v)
    {
        std::tie(idxx,idxy,score)=scoreInfo;
        scores.push_back(score);
        float xBias=coordsPtr[idxy*cw+idxx];
        float yBias=coordsPtr[cw*ch+idxy*cw+idxx];

        float xCoord=idxx*cellSize+cellStep+xBias*(crossRatio*cellStep);
        float yCoord=idxy*cellSize+cellStep+yBias*(crossRatio*cellStep);


        cv::KeyPoint kp(xCoord,yCoord,1.0);
        kps.push_back(kp);
        int featIdxxl=std::floor(xCoord/featCellSize);
        int featIdxxr=std::floor(xCoord/featCellSize)+1;

        int featIdxyl=std::floor(yCoord/featCellSize)*featCellSize;
        int featIdxyr=std::floor(yCoord/featCellSize)+1;

        float* descPtr= descriptors.ptr<float>(currentKpIdx);
        for (int i=0;i<fc;++i)
        {
            float x1=featIdxxl*featCellSize;
            float x2=featIdxxr*featCellSize;
            float y1=featIdxyl*featCellSize;
            float y2=featIdxyr*featCellSize;

            float f11=featPtr[i*fw*fh+featIdxyl*fw+featIdxxl];
            float f21=featPtr[i*fw*fh+featIdxyr*fw+featIdxxl];
            float f12=featPtr[i*fw*fh+featIdxyl*fw+featIdxxr];
            float f22=featPtr[i*fw*fh+featIdxyr*fw+featIdxxr];
            descPtr[i]=BilinearInter(xCoord, yCoord, featIdxxl, featIdxyl, featIdxxr, featIdxyr,  f11, f21, f12, f22);
        }
        ++currentKpIdx;
    }
    descs=descriptors;
    return true;

}


bool KP2D::Blob2Mat(const Blob::Ptr& blob,cv::Mat& blobMat){
    SizeVector blobSize = blob->getTensorDesc().getDims();
    std::vector<int> sizeTemp(blobSize.begin(),blobSize.end());
    cv::Mat result(sizeTemp,CV_32F);

    const float* blob_data=GetBlobReaderPtr<float>(blob);
    if(!blob_data)
    {
        return false;
    }
    for (int i=0;i<result.total();++i)
    {
        result.ptr<float>(0)[i]=blob_data[i];
    }
    blobMat=result;

    return true;
}

template<typename T>
const T* KP2D::GetBlobReaderPtr(const Blob::Ptr& blob)
{
    MemoryBlob::Ptr mblob=InferenceEngine::as<MemoryBlob>(blob);
    if(!mblob){
        return nullptr;
    }
    auto const mblobHolder=mblob->rmap();
    const T* blob_data=mblobHolder.as<const T *>();
    return blob_data;
}


template<typename T>
T* KP2D::GetBlobWritePtr(Blob::Ptr& blob)
{
    MemoryBlob::Ptr mblob=InferenceEngine::as<MemoryBlob>(blob);
    if(!mblob){
        return nullptr;
    }
    auto mblobHolder=mblob->wmap();
    T* blob_data=mblobHolder.as<T *>();
    return blob_data;
}

inline float KP2D::BilinearInter(float x,float y,float x1,float y1,float x2,float y2,float f11,float f21,float f12,float f22)
{
    return (f11*(x2-x)*(y2-y))/((x2-x1)*(y2-y1))+(f21*(x-x1)*(y2-y))/((x2-x1)*(y2-y1))+(f12*(x2-x)*(y-y1))/((x2-x1)*(y2-y1))+(f22*(x-x1)*(y-y1))/((x2-x1)*(y2-y1));
}
