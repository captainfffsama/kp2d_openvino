#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <string>

#include "inference_engine.hpp"
#include "common.hpp"
#include "ocv_common.hpp"

InferenceEngine::InferRequest InitModel(const std::string& modelPath){
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network = ie.ReadNetwork(modelPath);

    InferenceEngine::InputInfo::Ptr inputInfo = network.getInputsInfo().begin()->second;
    std::string input_name=network.getInputsInfo().begin()->first;

    std::cout << "input name is : " <<input_name <<std::endl;
    std::cout << "input info is : "<<inputInfo->getLayout() <<std::endl;

    for( auto ele : network.getOutputsInfo()){
        std::cout<< "output name is:" << ele.first<<std::endl;
        
    }

    InferenceEngine::ExecutableNetwork executableNetwork=ie.LoadNetwork(network,"CPU");
    InferenceEngine::InferRequest inferRequest=executableNetwork.CreateInferRequest();
    return inferRequest;
}


int main(int argc, char *argv[]) {
    std::string testImgPath="/home/chiebotgpuhq/pic_tmp/xianlan.jpeg";
    std::string modelPath="/home/chiebotgpuhq/intel/openvino_2021.2.185/deployment_tools/kp2d.xml";
    cv::Mat testImg=cv::imread(testImgPath);
    InferenceEngine::Blob::Ptr imgBlob=wrapMat2Blob(testImg);

    InferenceEngine::InferRequest inferRequest=InitModel(modelPath);

}
