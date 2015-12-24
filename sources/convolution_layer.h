#ifdef _MSC_VER
#pragma once
#endif

#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_


#include <vector>

#include "random.h"
#include "abstract_layer.h"

class ConvolutionLayer : public AbstractLayer {
private:

    static const int UNCONNECTED = 0;
    static const int CONNECTED = 1;

    struct Edge{
        int to = 0;
        int kernel_id = 0;
        int ky = 0;
        int kx = 0;

        Edge(int to_, int kernel_id_, int ky_, int kx_)
            : to(to_)
            , kernel_id(kernel_id_)
            , ky(ky_)
            , kx(kx_) {
        }
    };

private:
    Random* rng_ = nullptr;
    Size input_size_ = {};
    Size kernel_size_ = {};
    Size output_size_ = {};
    int n_featmap_in = 0;
    int n_featmap_out = 0;
    std::vector<std::vector<int> > connection_table_;

    std::vector<std::vector<Edge> > edge_io = {};
    std::vector<std::vector<Edge> > edge_oi = {};

    std::vector<Matrix> kernels = {};
    std::vector<double> bias = {};
    std::vector<Matrix> dkernels = {};
    std::vector<double> dbias = {};

public:
    ConvolutionLayer(Random* rng, Size input_size, Size kernel_size, int featmap_in, int featmap_out, const std::vector<std::vector<int> >& connection_table = {});
    virtual ~ConvolutionLayer();

    const Matrix& forward_propagation(const Matrix& input) override;
    Matrix back_propagation(const Matrix& error, double eta = 0.1, double momentum = 0.5) override;
    
private:
    void initialize();
    void add_edge(int input_id, int output_id, int kernel_id, int ky, int kx);
};

#endif  // _CONVOLUTION_LAYER_H_
