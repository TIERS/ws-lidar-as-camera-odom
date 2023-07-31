#ifndef __NETWORK_H_
#define __NETWORK_H_

#include <torch/torch.h>

class SuperPointNet : public torch::nn::Module {
public:
    SuperPointNet();

    std::vector<torch::Tensor> forward(torch::Tensor x);

private:
    const int c1 = 64, c2 = 64, c3 = 128, c4 = 128, c5 = 256, d1 = 256;

    torch::nn::ReLU relu_;
    torch::nn::MaxPool2d pool_;
    
    torch::nn::Conv2d conv1a_;
    torch::nn::Conv2d conv1b_;
    torch::nn::Conv2d conv2a_;
    torch::nn::Conv2d conv2b_;
    torch::nn::Conv2d conv3a_;
    torch::nn::Conv2d conv3b_;
    torch::nn::Conv2d conv4a_;
    torch::nn::Conv2d conv4b_;

    torch::nn::Conv2d convPa_;
    torch::nn::Conv2d convPb_;

    torch::nn::Conv2d convDa_;
    torch::nn::Conv2d convDb_;

};
#endif
