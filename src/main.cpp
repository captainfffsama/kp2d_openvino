#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

#include "inference_engine.hpp"
#include "utils/ocv_common.hpp"


void InitModel(const std::string& modelPath, InferenceEngine::InferRequest& inferRequest, std::string& inputName, std::vector<std::string>& outputNames){
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(modelPath);
    auto input_shape=network.getInputShapes();
    std::cout<<input_shape.begin()->first<<std::endl;

    InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
    inputName=network.getInputsInfo().begin()->first;
    
    inputInfo->setLayout(InferenceEngine::Layout::NHWC);
    inputInfo->setPrecision(InferenceEngine::Precision::FP32);

    for( auto ele : network.getOutputsInfo()){
        outputNames.push_back(ele.first);
        ele.second->setPrecision(InferenceEngine::Precision::FP32);
    }

    InferenceEngine::ExecutableNetwork executableNetwork=ie.LoadNetwork(network,"CPU");
    inferRequest=executableNetwork.CreateInferRequest();
}


int SizeLength(const std::vector<int>& size){
    int result=1;
    for(auto i:size){
        result *= i;
    }
    return result;
}


/*!
 * @brief 用于将32的blob转为32的Mat
 *
 * @param blob 需要转的blob
 * @param blobMat 输出的Mat,一般就是Mat(),里面会再次初始化
 * @param precision 用于指示精度,后面要改
 */
void Blob2Mat(const std::shared_ptr<InferenceEngine::Blob>& blob,cv::Mat blobMat,uint8_t precision=CV_32F){
    // TODO: 这里先写死用float 后面再改
    auto shape = blob->getTensorDesc().getDims();
    std::vector<int> matShape (begin(shape),end(shape));
    blobMat.create(matShape,precision);
    InferenceEngine::MemoryBlob::Ptr mblob=InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    auto mblobHolder=mblob->rmap();
    float* blob_data=mblobHolder.as<float*>();
    auto mat_ptr=blobMat.ptr<float>();

    for (int i=0;i<blob->size();i++){
        mat_ptr[i]=blob_data[i];
    }

}



int main(int argc, char *argv[]) {
    std::clock_t start,end;
    std::string testImgPath="/home/chiebotgpuhq/pic_tmp/xianlan1.jpeg";
    std::string modelPath="/home/chiebotgpuhq/intel/openvino_2021.2.185/deployment_tools/kp2d.xml";
    cv::Mat testImg=cv::imread(testImgPath);
    std::cout<<testImg.type()<<std::endl;
    cv::normalize(testImg,testImg,1,0,cv::NORM_INF);
    std::cout<<"final mat type:"<<testImg<<std::endl;

    
    InferenceEngine::Blob::Ptr imgBlob=wrapMat2Blob(testImg);

    InferenceEngine::InferRequest inferRequest;
    std::string inputName="";
    std::vector<std::string> outputNames;
    start=clock();
    InitModel(modelPath,inferRequest,inputName,outputNames);
    end=clock();
    std::cout<< "init spend time:" << (double)(end-start)/CLOCKS_PER_SEC<< std::endl;

    start=clock();
    inferRequest.SetBlob(inputName,imgBlob);
    std::vector<InferenceEngine::Blob::Ptr> outputs;
    for(auto name:outputNames){
        InferenceEngine::Blob::Ptr out=inferRequest.GetBlob(name);
        outputs.push_back(out);
        std::cout<< name <<":"<<out->size()<<std::endl;
        for(auto e:out->getTensorDesc().getDims())
        {
            std::cout<<e<<" ";
        }
        std::cout<<std::endl;
    }
    end=clock();
    std::cout<< "infer spend time:" << (double)(end-start)/CLOCKS_PER_SEC<< std::endl;

    auto score=inferRequest.GetBlob("score");
    cv::Mat scoreMat=cv::Mat();
    Blob2Mat(score, scoreMat);
    std::cout<<"score col"<<scoreMat.cols<< std::endl;


    cv::Mat a=cv::imread("/home/chiebotgpuhq/pic_tmp/xianlan.jpeg");
    cv::Mat b= a/2;

    cv::Mat c;
    start=clock();
    for (int i=0;i<256;i++)
    {
        cv::resize(a, c, cv::Size(1920, 1080), 0,0);
    }
    end=clock();
    std::cout<<"resize spend time:" << (double)(end-start)/CLOCKS_PER_SEC<< std::endl;
    std::cout<<c.size<<std::endl;


    
    
    
    

}
