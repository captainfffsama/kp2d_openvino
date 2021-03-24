#include "kp2d.hpp"
#include "opencv2/opencv.hpp"
#include <cstdint>
#include <tuple>

using namespace kp2d;
using namespace InferenceEngine;

KP2D::KP2D(const std::string& xmlPath,int downSample){
    network=ie.ReadNetwork(xmlPath);
    cellSize=downSample;

}

KP2D::~KP2D(){
}

void KP2D::setInputInfo(){
    for(auto& ele:network.getInputsInfo()){
        auto& inputData=ele.second;
        inputData->setPrecision(Precision::FP32);
        inputData->setLayout(Layout::NCHW);
        inputData->getPreProcess().setColorFormat(ColorFormat::BGR);
    }
}

void KP2D::setOutputInfo(){
    for(auto& ele:network.getOutputsInfo()){
        auto& outData=ele.second;
        outData->setPrecision(Precision::FP32);
        outData->setLayout(Layout::NCHW);
    }
}

void KP2D::infer(const cv::Mat &src, KPResult &kpresult){
    auto originSize=src.size();
    cv::Size newSize;
    Blob::Ptr srcBlob;
    preProcess(src, srcBlob,newSize);
    std::cout<<newSize<<std::endl;
    // newSize.height=1088;
    // newSize.width=1928;

    //  根据图片尺寸调整网络
    //  TODO: 后续可以考虑加个dynamic参数,对于固定的图片就固定住网络
    auto inputShapes=network.getInputShapes();
    //std::string inputName;
    SizeVector inputShape {1,3,static_cast<unsigned long>(newSize.height),static_cast<unsigned long>(newSize.width)};
    inputShapes["x"]=inputShape;

    network.reshape(inputShapes);
    setInputInfo();
    setOutputInfo();
    exeNetwork=ie.LoadNetwork(network,"CPU");
    inferRequest=exeNetwork.CreateInferRequest();

    inferRequest.SetBlob("x",srcBlob);
    Blob::Ptr coordBlob=inferRequest.GetBlob("coord");
    Blob::Ptr scoreBlob=inferRequest.GetBlob("score");
    Blob::Ptr featBlob=inferRequest.GetBlob("feat");
    for (auto e:coordBlob->getTensorDesc().getDims()){
        std::cout<<e<<std::endl;
    }

}

void KP2D::preProcess(const cv::Mat& src, Blob::Ptr& srcBlob,cv::Size& newSize){
    cv::Mat dst;
    if (5!=(0x0F & src.type())){
        src.convertTo(dst,CV_32F);
    }
    else{
        dst=src.clone();
    }
    cv::normalize(dst,dst,1,0,cv::NORM_INF);
    int bottomPad=(cellSize-dst.size().height%cellSize)%cellSize;
    int rightPad=(cellSize-dst.size().width%cellSize)%cellSize;
    cv::copyMakeBorder(dst, dst, 0, bottomPad, 0, rightPad, cv::BORDER_CONSTANT|cv::BORDER_ISOLATED,cv::Scalar::all(0));
    srcBlob=warpMat2Blob(dst);
    newSize=dst.size();
    
}

Blob::Ptr KP2D::warpMat2Blob(const cv::Mat& mat){
    size_t channels = mat.channels();
    size_t height = mat.size().height;
    size_t width = mat.size().width;


    if (!mat.isContinuous()) THROW_IE_EXCEPTION
                << "Doesn't support conversion from not dense cv::Mat";

    TensorDesc tDesc(Precision::FP32,
                    {1, channels, height, width},
                    Layout::NHWC);
    
    std::cout<<mat.type()<<std::endl;
    std::cout<<tDesc.getPrecision().hasStorageType<float>()<<std::endl;
    return make_shared_blob<float>(tDesc, reinterpret_cast<float*>(mat.data));
}
