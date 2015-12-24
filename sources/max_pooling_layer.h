#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MAX_POOLING_LAYER_H_
#define _MAX_POOLING_LAYER_H_

#include <vector>

#include "random.h"
#include "abstract_layer.h"

class MaxPoolingLayer : public AbstractLayer {
private:
    struct Edge {
        int to = 0;
        int featmap = 0;
        int rev = 0;
        bool active = false;

        Edge(int to_, int featmap_, int rev_)
            : to(to_)
            , featmap(featmap_)
            , rev(rev_) {
        }
    };

private:
    Random* rng_ = nullptr;
    Size input_size_ = {};
    Size pool_size_ {};
    Size output_size_ = {};
    int n_featmap_ = 0;

    std::vector<std::vector<Edge> > edge_io = {};
    std::vector<std::vector<Edge> > edge_oi = {};
    std::vector<double> scale = {};
    std::vector<double> bias = {};
    std::vector<double> dscale = {};
    std::vector<double> dbias = {};

public:
    MaxPoolingLayer(Random* rng, Size input_size, Size pool_size, int n_featmap);
    virtual ~MaxPoolingLayer();

    const Matrix& forward_propagation(const Matrix& input) override;
    Matrix back_propagation(const Matrix& error, double eta = 0.1, double momentum = 0.5) override;

private:
    void initialize();
    void add_edge(int in, int out, int f);
};

#endif  // _MAX_POOLING_LAYER_H_
