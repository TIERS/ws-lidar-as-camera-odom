#include "superpoint/SuperPointFrontend.h"

SPFrontend::SPFrontend(std::string weight_path, float nms_dist, float conf_thresh, float nn_thresh)
    : nms_dist_(nms_dist), conf_thresh_(conf_thresh), nn_thresh_(nn_thresh)
{
    model = std::make_shared<SuperPointNet>();
    torch::load(model, weight_path);
    torch::autograd::GradMode::set_enabled(false);

    model->eval();
}

// std::vector<cv::KeyPoint> SPFrontend::run(cv::Mat img)
void SPFrontend::run(cv::Mat img, std::vector<cv::KeyPoint>& filtered_kpoints, cv::Mat& descMat_out)
{
    int H = img.rows;
    int W = img.cols;
    auto img_c = img.clone();
    auto inp = torch::from_blob(img_c.data, {1, 1, H, W}, torch::kFloat32);
    if (cuda)
    {
        inp = inp.to(torch::kCUDA);
    }
    inp = inp.to(torch::kCPU);
    auto outs = model->forward({inp});
    torch::Tensor semi = outs[0];
    torch::Tensor coarse_desc = outs[1];

    semi = semi.squeeze();                    // [65 , H , W]
    torch::Tensor dense = semi.exp();         // softmax
    dense = dense / (dense.sum(0) + 0.00001); // sum to 1

    // remove dustbin
    torch::Tensor nodust = dense.slice(0, 0, 64);

    int Hc = int(H / cell_);
    int Wc = int(W / cell_);
    nodust = nodust.permute({1, 2, 0}); // [60, 80, 64]

    torch::Tensor heatmap = nodust.reshape({Hc, Wc, cell_, cell_});
    heatmap = heatmap.permute({0, 2, 1, 3});
    heatmap = heatmap.reshape({Hc * cell_, Wc * cell_});

    auto kpts = (heatmap >= conf_thresh_);
    auto kpts_nonzero = torch::nonzero(kpts);
    kpts_nonzero = kpts_nonzero.transpose(1, 0); // [2, ]
    auto xs = kpts_nonzero[0];
    auto ys = kpts_nonzero[1];
    auto prob = torch::zeros({xs.size(0)});
    for (int i = 0; i < xs.size(0); i++)
    {
        torch::Scalar x = xs[i].item();
        torch::Scalar y = ys[i].item();
        prob.slice(0, i, i + 1) = heatmap[x][y];
    }
    torch::Tensor pts = torch::zeros({3, xs.size(0)});
    pts[0] = ys;
    pts[1] = xs;
    pts[2] = prob;

    auto pts_out = nms_fast(pts[0], pts[1], pts[2], H, W, nms_dist_);

    // 重新排序(按照得分降序排序)
    std::vector<int> inds(pts_out.size());
    for (int i = 0; i < inds.size(); i++)
    {
        inds[i] = i;
    }
    std::sort(inds.begin(), inds.end(), [&](int i, int j)
              { return pts_out[i].response > pts_out[j].response; });

    // // Remove points along border
    int bord = border_remove;
    // std::vector<std::pair<cv::Point2d, double>> filtered_kpoints;
    // std::vector<cv::KeyPoint> filtered_kpoints;
    for (int i = 0; i < pts_out.size(); i++)
    {
        if (pts_out[i].pt.x >= bord && pts_out[i].pt.x < (W - bord) &&
            pts_out[i].pt.y >= bord && pts_out[i].pt.y < (H - bord))
        {
            filtered_kpoints.push_back(pts_out[i]);
        }
    }

    // -- Descirptor --
    torch::Tensor desc;
    int D = coarse_desc.size(1);
    // cv::Mat descMat_out;
    if (pts_out.size() == 0)
    {
        descMat_out = cv::Mat::zeros(D, 0, CV_32F);
    }
    else
    {
        std::vector<cv::Point2f> temp;
        for (const auto &kp : pts_out)
        {
            temp.push_back(kp.pt);
        }
        torch::Tensor samp_pts = torch::from_blob(temp.data(), {static_cast<int32_t>(temp.size()), 2}, torch::kFloat);
        samp_pts = samp_pts.permute({1, 0});
        float w_scale = 2.0 / static_cast<float>(W);
        float h_scale = 2.0 / static_cast<float>(H);
        samp_pts[0] = (samp_pts[0] * w_scale) - 1.0;
        samp_pts[1] = (samp_pts[1] * h_scale) - 1.0;
        samp_pts = samp_pts.transpose(0, 1);
        if (!samp_pts.is_contiguous())
        {
            samp_pts = samp_pts.contiguous();
        }
        samp_pts = samp_pts.view({1, 1, -1, 2}); // [1, 1, N, 2]
        samp_pts = samp_pts.to(torch::kFloat);
        if (cuda)
        {
            samp_pts = samp_pts.to(torch::kCUDA);
        }
        torch::nn::functional::GridSampleFuncOptions options;
        options.mode(torch::kBilinear);
        options.padding_mode(torch::kZeros);
        options.align_corners(true);

        desc = torch::nn::functional::grid_sample(coarse_desc, samp_pts, options);
        desc = desc.to(torch::kCPU);
        // 获取数据的指针
        float* data_ptr = desc.data_ptr<float>();
        // 张量维度信息
        torch::IntArrayRef sizes = desc.sizes();
        std::vector<int32_t> dims(sizes.size());
        std::copy(sizes.begin(), sizes.end(), dims.begin());
        cv::Mat descMat(dims[1], dims[3], CV_32F);
        for (int32_t r = 0; r < dims[1]; r++)
        {
            for (int32_t c = 0; c < dims[3]; c++)
            {
                float value = data_ptr[r * dims[3] + c];
                descMat.at<float>(r, c) = value;
            }
        }
        
        // 计算 descMat 每列的 L2 范数
        std::vector<double> norms;
        for (int c = 0; c < descMat.cols; c++)
        {
            cv::Mat column = descMat.col(c);
            double norm = cv::norm(column, cv::NORM_L2);
            norms.push_back(norm);
        }
        // 更新 descMat
        for (int r = 0; r < descMat.rows; r++)
        {
            for (int c = 0; c < descMat.cols; c++)
            {
                float value = descMat.at<float>(r, c);
                descMat.at<float>(r, c) = value / norms[c]; 
            }
        }
        descMat_out = descMat;
    }
    
}


std::vector<cv::KeyPoint> SPFrontend::nms_fast(torch::Tensor xs, torch::Tensor ys, torch::Tensor prob, int H, int W, float dist_thresh)
{
    // Track NMS data.
    cv::Mat grid = cv::Mat::zeros(H, W, CV_32S);
    // Store indices of points.
    cv::Mat inds = cv::Mat::zeros(H, W, CV_32S);
    // Sort corners by confidence and round to nearest integer.
    std::vector<int> sorted_inds(xs.size(0));
    for (int i = 0; i < xs.size(0); i++)
    {
        sorted_inds[i] = i;
    }
    // 对vector<int> 中的向量sorted_inds进行排序，规则按照 prob 张量中元素大小进行降序排序
    std::sort(sorted_inds.begin(), sorted_inds.end(), [&](int a, int b)
              { return prob[a].item<double>() > prob[b].item<double>(); });
    std::vector<cv::KeyPoint> rcorners(xs.size(0));
    for (int i = 0; i < xs.size(0); i++)
    {
        rcorners[i].pt.x = std::round(xs[sorted_inds[i]].item<double>());
        rcorners[i].pt.y = std::round(ys[sorted_inds[i]].item<double>());
        rcorners[i].response = std::round(prob[sorted_inds[i]].item<double>());
    }
    // Check for edge case of 0 or 1 corners
    if (rcorners.size() == 0)
    {
        return std::vector<cv::KeyPoint>();
    }
    if (rcorners.size() == 1)
    {
        std::vector<cv::KeyPoint> out;
        out.reserve(1);
        cv::KeyPoint kpt;
        kpt.pt = rcorners[0].pt;

        out.push_back(kpt);
        return out;
    }
    // Initialize the grid
    // 把检测到的点置成 1
    for (int i = 0; i < rcorners.size(); i++)
    {
        grid.at<int>(rcorners[i].pt.y, rcorners[i].pt.x) = 1;
        inds.at<int>(rcorners[i].pt.y, rcorners[i].pt.x) = i;
    }
    // Pad the border of grid, so that we can NMS points near the border
    int pad = std::ceil(dist_thresh);
    cv::Mat grid_pad;
    cv::copyMakeBorder(grid, grid_pad, pad, pad, pad, pad, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Iterate through points, highest to lowest conf, suppress neighborhood
    int count = 0;
    for (int i = 0; i < rcorners.size(); i++)
    {
        cv::Point2i pt(rcorners[i].pt.x + pad, rcorners[i].pt.y + pad);
        if (grid_pad.at<int>(pt.y, pt.x) == 1)
        {
            cv::Rect rect(pt.x - pad, pt.y - pad, 2 * pad + 1, 2 * pad + 1);
            grid_pad(rect) = 0;
            grid_pad.at<int>(pt.y, pt.x) = -1;
            count++;
        }
    }
    std::vector<int> keepx;
    std::vector<int> keepy;
    for (int r = 0; r < grid_pad.rows; r++)
    {
        for (int c = 0; c < grid_pad.cols; c++)
        {
            // std::cout << r  << std::endl;
            if (grid_pad.at<int>(r, c) == -1)
            {
                keepy.push_back(r - pad);
                keepx.push_back(c - pad);
            }
        }
    }
    // --------------------------------------------------------
    std::vector<int> inds_keep;
    for (int i = 0; i < keepx.size(); i++)
    {
        int cur;
        cur = inds.at<int>(keepy[i], keepx[i]);
        inds_keep.push_back(cur);
    }
    std::vector<cv::KeyPoint> out;
    for (int i = 0; i < inds_keep.size(); i++)
    {
        cv::KeyPoint kpoint;
        kpoint.pt.x = xs[sorted_inds[inds_keep[i]]].item<double>();
        kpoint.pt.y = ys[sorted_inds[inds_keep[i]]].item<double>();
        kpoint.response = prob[sorted_inds[inds_keep[i]]].item<double>();

        out.push_back(kpoint);
    }

    std::vector<double> values;
    for (int i = 0; i < out.size(); i++)
    {
        double cur;
        cur = out[i].response;
        values.push_back(cur);
    }
    // 初始化排序数组
    std::vector<int> sorted_inds2(values.size());
    for (int i = 0; i < values.size(); i++)
    {
        sorted_inds2[i] = i;
    }
    // 降序排序
    std::sort(sorted_inds2.begin(), sorted_inds2.end(), [&](size_t a, size_t b)
              { return values[a] > values[b]; });

    std::vector<cv::KeyPoint> sorted_out;
    for (const auto &index : sorted_inds2)
    {
        sorted_out.push_back(out[index]);
    }

    // 根据排序后的索引值 sorted_inds2 , 从 inds_keep 数组中选择相应的元素, 并将结果存储在 out_inds 中
    // 1 inds_keep 是根据 keepx 和 keepy 的值从 inds 数组中获取的索引，表示经过非极大值抑制后保留的点的原始索引
    // 2 sorted_inds2 是根据 values 的降序排列得到的索引，表示按照置信度从高到低排序后点的坐标
    // 3 out_inds 是通过将 inds_keep 数组按照 sorted_inds2 的顺序重新排序i得到的，目的是将经过非极大值抑制和
    // 排序后的点的索引与原始的点的索引对应起来，方便后续操作
    // std::vector<int> out_inds;
    // out_inds.reserve(sorted_inds2.size());
    // for (int i = 0; i < sorted_inds2.size(); i++)
    // {
    //     int idx = inds_keep[sorted_inds2[i]];
    //     out_inds.push_back(sorted_inds[idx]);
    // }
    return sorted_out;
}