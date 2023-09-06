# include "superpoint/NetWork.hpp"


SuperPointNet::SuperPointNet()
    :   relu_(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        pool_(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))),

        // Shared Encoder.
        conv1a_(torch::nn::Conv2d(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1))),
        conv1b_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1))),
        conv2a_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1))),
        conv2b_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1))),
        conv3a_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1))),
        conv3b_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1))),
        conv4a_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1))),
        conv4b_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1))),

        // Detector Head.
        convPa_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1))),
        convPb_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0))),

        // Descriptor Head.
        convDa_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1))),
        convDb_(torch::nn::Conv2d(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0)))
{
    register_module("relu", relu_);
    register_module("pool", pool_);
    register_module("conv1a", conv1a_);
    register_module("conv1b", conv1b_);
    register_module("conv2a", conv2a_);
    register_module("conv2b", conv2b_);
    register_module("conv3a", conv3a_);
    register_module("conv3b", conv3b_);
    register_module("conv4a", conv4a_);
    register_module("conv4b", conv4b_);
    register_module("convPa", convPa_);
    register_module("convPb", convPb_);
    register_module("convDa", convDa_);
    register_module("convDb", convDb_);
}

std::vector<torch::Tensor> SuperPointNet::forward(torch::Tensor x)
{

    // Shared Encoder
    x = relu_(conv1a_->forward(x));
    x = relu_(conv1b_->forward(x));
    x = pool_->forward(x);
    x = relu_(conv2a_->forward(x));
    x = relu_(conv2b_->forward(x));
    x = pool_->forward(x);
    x = relu_(conv3a_->forward(x));
    x = relu_(conv3b_->forward(x));
    x = pool_->forward(x);
    x = relu_(conv4a_->forward(x));
    x = relu_(conv4b_->forward(x));

    // Detector Head.
    torch::Tensor cPa = relu_(convPa_->forward(x));
    torch::Tensor semi = convPb_->forward(cPa); // [N, 65, H/8, W/8]

    // Descriptor Head.
    torch::Tensor cDa = relu_(convDa_->forward(x));
    torch::Tensor desc = convDb_->forward(cDa); // [N, 256, H/8, W/8]

    auto dn = torch::norm(desc, 2, 1);
    desc = desc.div(torch::unsqueeze(dn, 1));

    std::vector<torch::Tensor> ret;
    ret.push_back(semi);
    ret.push_back(desc);

    return ret;
}
