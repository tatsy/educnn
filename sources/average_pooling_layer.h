#ifdef _MSC_VER
#pragma once
#endif

#ifndef _AVERAGE_POOLING_LAYER_H_
#define _AVERAGE_POOLING_LAYER_H_

#include <vector>

#include "random.h"
#include "abstract_layer.h"

class AveragePoolingLayer : public AbstractLayer {
private:
    struct Edge {
        int to = 0;
        int featmap = 0;

        Edge(int to_, int featmap_)
            : to(to_)
            , featmap(featmap_) {
        }
    };

private:
    Random* rng_ = nullptr;
    Size input_size_ = {};
    Size pool_size_{};
    Size output_size_ = {};
    int n_featmap_ = 0;

    std::vector<std::vector<Edge> > edge_io = {};
    std::vector<std::vector<Edge> > edge_oi = {};
    std::vector<double> scale = {};
    std::vector<double> bias = {};
    std::vector<double> dscale = {};
    std::vector<double> dbias = {};

public:
    AveragePoolingLayer(Random* rng, Size input_size, Size pool_size, int n_featmap);

    virtual ~AveragePoolingLayer();
    
    const Matrix& forward_propagation(const Matrix& input) override;
    Matrix back_propagation(const Matrix& error, double eta = 0.1, double momentum = 0.5) override;

private:
    void initialize();
    void add_edge(int in, int out, int f);    
};

#endif  // _AVERAGE_POOLING_LAYER_H_
