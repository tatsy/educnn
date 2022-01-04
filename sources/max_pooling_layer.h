#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MAX_POOLING_LAYER_H_
#define _MAX_POOLING_LAYER_H_

#include <vector>

#include "openmp.h"
#include "random.h"
#include "abstract_layer.h"

class MaxPoolingLayer : public AbstractLayer {
public:
    // Public methods
    MaxPoolingLayer(Size input_size, Size pool_size,
                    int n_featmap)
        : AbstractLayer()
        , input_size_(input_size)
        , pool_size_(pool_size)
        , output_size_()
        , n_featmap_(n_featmap) {
        Assertion(input_size_.rows % pool_size_.rows == 0 &&
                    input_size_.cols % pool_size_.cols == 0,
                    "input size and pool size are not compatible!!");

        output_size_.rows = input_size_.rows / pool_size_.rows;
        output_size_.cols = input_size_.cols / pool_size_.cols;

        edge_io.resize(input_size_.total() * n_featmap);
        edge_oi.resize(output_size_.total() * n_featmap);

        initialize();
    }

    virtual ~MaxPoolingLayer() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        const int batchsize = input.rows();
        const int n_output = edge_oi.size();
        
        input_ = input;
        output_ = Matrix::Zero(batchsize, n_output);

        omp_parallel_for (int b = 0; b < batchsize; b++) {
            for (int o = 0; o < n_output; o++) {
                double maxval = -INFTY;
                int active_index = 0;
                for (int e = 0; e < edge_oi[o].size(); e++) {
                    Edge& edge = edge_oi[o][e];
                    if (maxval < input(b, edge.to)) {
                        maxval = input(b, edge.to);
                        active_index = e;
                    }
                }

                Edge& active_e = edge_oi[o][active_index];
                output_(b, o) = input(b, active_e.to);
            }
        }

        return output_;    
    }

    Matrix back_propagation(const Matrix& dLdy, double eta = 0.1,
                            double momentum = 0.5) override {
        const int batchsize = dLdy.rows();
        const int n_input = input_size_.total() * n_featmap_;
        const int n_output = output_size_.total() * n_featmap_;
        Matrix dLdx = Matrix::Zero(batchsize, n_input);
        omp_parallel_for (int b = 0; b < batchsize; b++) {
            for (int o = 0; o < n_output; o++) {
                double maxval = -INFTY;
                int active_index = 0;
                for (int e = 0; e < edge_oi[o].size(); e++) {
                    Edge& edge = edge_oi[o][e];
                    if (maxval < input_(b, edge.to)) {
                        maxval = input_(b, edge.to);
                        active_index = e;
                    }
                }

                Edge& active_e = edge_oi[o][active_index];
                dLdx(b, active_e.to) += dLdy(b, o);
            }       
        }

        return dLdx;
    }

private:
    // Private methods
    void initialize() {
        for (int f = 0; f < n_featmap_; f++) {
            for (int yout = 0; yout < output_size_.rows; yout++) {
                for (int xout = 0; xout < output_size_.cols; xout++) {
                    initialize_edges(f, yout, xout);
                }
            }
        }
    }

    void initialize_edges(int f, int yout, int xout) {
        for (int dy = 0; dy < pool_size_.rows; dy++) {
            for (int dx = 0; dx < pool_size_.cols; dx++) {
                int yin = yout * pool_size_.rows + dy;
                int xin = xout * pool_size_.cols + dx;

                int input_index = f * input_size_.total() + 
                                  (yin * input_size_.cols + xin);
                int output_index = f * output_size_.total() +
                                   (yout * output_size_.cols + xout);
                add_edge(input_index, output_index, f);
            }
        }        
    }

    void add_edge(int in, int out, int f) {
        int n_io = edge_io[in].size();
        int n_oi = edge_oi[out].size();
        edge_io[in].emplace_back(out, f, n_oi);
        edge_oi[out].emplace_back(in, f, n_io);    
    }

    // Private internal classes
    struct Edge {
        int to = 0;
        int rev = 0;

        Edge(int to_, int featmap_, int rev_)
            : to(to_)
            , rev(rev_) {
        }
    };

    // Private parameters
    Size input_size_ = {};
    Size pool_size_ {};
    Size output_size_ = {};
    int n_featmap_ = 0;

    std::vector<std::vector<Edge> > edge_io = {};
    std::vector<std::vector<Edge> > edge_oi = {};

}; // class MaxPoolingLayer

#endif  // _MAX_POOLING_LAYER_H_
