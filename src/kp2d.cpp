#include "kp2d.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <tuple>

using namespace kp2d;
using namespace InferenceEngine;

KP2D::KP2D(const std::string& xmlPath,int top_k,float scoreThr,int downSample,int featLength,int batch)
    :cellSize(downSample),topK(top_k),threshold(scoreThr),scoreDim(1),featDim(featLength),coordDim(2),batchSize(batch)
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
        std::cout<<inputData->getLayout()<<std::endl;
    }
}

void KP2D::SetOutputInfo(){
    for(auto& ele:network.getOutputsInfo()){
        auto& outData=ele.second;
        outData->setPrecision(Precision::FP32);
        outData->setLayout(Layout::NCHW);
    }
}

bool KP2D::Infer(const cv::Mat &src, KPResult &kpresult){
    auto originSize=src.size();
    Blob::Ptr srcBlob=PreProcess(src);

    
    if (!srcBlob)
    {
        return false;
    }

    //  根据图片尺寸调整网络
    //  TODO: 后续可以考虑加个dynamic参数,对于固定的图片就固定住网络


    inferRequest.SetBlob("x",srcBlob);
    Blob::Ptr coordBlob=inferRequest.GetBlob("coord");
    Blob::Ptr scoreBlob=inferRequest.GetBlob("score");
    Blob::Ptr featBlob=inferRequest.GetBlob("feat");

    std::cout<<"score blob precision:"<<scoreBlob->getTensorDesc().getPrecision()<<std::endl;

    if(!PostProcess(coordBlob,scoreBlob,featBlob,kpresult))
    {
        std::cout<<"ERROR!! post process fail!!"<<std::endl;
        return false;
    }
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

bool KP2D::PostProcess(const Blob::Ptr& coordsBlob, const Blob::Ptr& scoreBlob,const Blob::Ptr& featBlob, KPResult& result){
    const float* scorePtr=GetBlobReaderPtr<float>(scoreBlob);
    const float* featPtr=GetBlobReaderPtr<float>(featBlob);
    const float* coordsPtr=GetBlobReaderPtr<float>(coordsBlob);

    if (!((scorePtr && featPtr) && coordsPtr))
    {
        std::cout<<"ERROR!!! get data failed"<<std::endl;
        return false;
    }
    int kpNumberCount=topK;

    auto scoreSize=scoreBlob->getTensorDesc().getDims();
    int sw=scoreSize[3];
    int sh=scoreSize[2];
    std::cout<<sw<<std::endl;
    std::cout<<sh<<std::endl;

    int imgW=sw*cellSize;
    int imgH=sh*cellSize;

    std::vector<ScoreInfo> scoreIdxTemp_v;
    std::make_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end());

    int beyondS=0;
    int belowS=0;
    int zeroNum=0;

    for (int h=1;h<(sh-1);h++){
        for (int w=1;w<(sw-1);w++){
            if(scorePtr[h*sw+w]>1)
            {
                ++beyondS;
            }
            else if(scorePtr[h*sw+w] < 0)
            {
                ++belowS;
            }
            else if(scorePtr[h*sw+w] ==0)
            {++zeroNum;}
            else
            {
                scoreIdxTemp_v.push_back(std::make_tuple(w,h,scorePtr[h*sw+w]));
                std::push_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end(),[](const ScoreInfo& A,const ScoreInfo& B) {return std::get<2>(A) < std::get<2>(B); });

            }
        }
    }

    std::cout<<" 超过1的分数有:"<<beyondS<<std::endl;
    std::cout<<" 低于0的分数有:"<<belowS<<std::endl;
    std::cout<<" 等于0的分数有:"<<zeroNum<<std::endl;
    std::cout<<" 正常范围分数有:"<<scoreIdxTemp_v.size()<<std::endl;

    for (int i=0;i<scoreIdxTemp_v.size();++i)
    {
        std::pop_heap(scoreIdxTemp_v.begin(),scoreIdxTemp_v.end(),
                [](const ScoreInfo& A,const ScoreInfo& B) {return std::get<2>(A) < std::get<2>(B); });
        int idxx=0,idxy=0;
        float score=0;
        std::tie(idxx,idxy,score)=scoreIdxTemp_v.back();
        if(!i)
        {
            std::cout<<" "<<idxx<<" "<<idxy<<" "<<score<<std::endl;
        }
    }

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
