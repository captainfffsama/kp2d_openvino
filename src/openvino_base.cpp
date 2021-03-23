#include "openvino_base.hpp"

using namespace openvino;
using namespace InferenceEngine;

VinoBase::VinoBase(const std::string& xmlPath){
    network=ie.ReadNetwork(xmlPath);
    exeNetwork=ie.LoadNetwork(network,"CPU");
    inferRequest=exeNetwork.CreateInferRequest();

    setInputInfo();
    setOutputInfo();
}

VinoBase::~VinoBase(){
}

void VinoBase::setInputInfo(){
    for(auto& ele:network.getInputsInfo()){
        auto& inputData=ele.second;
        inputData->setPrecision(Precision::FP32);
        inputData->setLayout(Layout::NCHW);
        inputData->getPreProcess().setColorFormat(ColorFormat::BGR);
    }
}

void VinoBase::setOutputInfo(){
    for(auto& ele:network.getOutputsInfo()){
        auto& outData=ele.second;
        outData->setPrecision(Precision::FP32);
    }
}
