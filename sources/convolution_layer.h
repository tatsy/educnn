#ifdef _MSC_VER
#pragma once
#endif

#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_


#include <vector>

#include "random.h"
#include "abstract_layer.h"

class ConvolutionLayer : public AbstractLayer {
public:
    // Public methods
    ConvolutionLayer(Random* rng, Size input_size, Size kernel_size,
                     int featmap_in, int featmap_out,
                     const std::vector<std::vector<int>>& connect_table = {})
        : AbstractLayer()
        , rng_(rng)
        , input_size_(input_size)
        , kernel_size_(kernel_size)
        , output_size_()
        , n_featmap_in(featmap_in)
        , n_featmap_out(featmap_out)
        , connection_table_(connect_table) {
        output_size_.rows = input_size_.rows - kernel_size_.rows + 1;
        output_size_.cols = input_size_.cols - kernel_size_.cols + 1;

        edge_io.resize(input_size_.total() * n_featmap_in);
        edge_oi.resize(output_size_.total() * n_featmap_out);

        kernels.resize(n_featmap_in * n_featmap_out);
        bias.resize(n_featmap_in * n_featmap_out);
        dkernels.resize(n_featmap_in * n_featmap_out);
        dbias.resize(n_featmap_in * n_featmap_out);
        for (int k = 0; k < n_featmap_in * n_featmap_out; k++) {
            kernels[k] = Matrix(kernel_size_.rows, kernel_size_.cols);
            dkernels[k] = Matrix(kernel_size.rows, kernel_size_.cols);
            for (int i = 0; i < kernel_size_.rows; i++) {
                for (int j = 0; j < kernel_size_.cols; j++) {
                    kernels[k](i, j) = rng->normal() * 0.1;
                    dkernels[k](i, j) = 0.0;
                }
            }
            bias[k] = 0.0;
            dbias[k] = 0.0;
        }

        if (connection_table_.empty()) {
            connection_table_ = 
                std::vector<std::vector<int> >(
                    n_featmap_in, std::vector<int>(n_featmap_out, CONNECTED));
        }

        initialize();
    }

    virtual ~ConvolutionLayer() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        int n_data = input.cols();
        int n_output = edge_oi.size();

        input_ = input;
        output_ = Matrix::Zero(n_output, n_data);
        for (int i = 0; i < n_data; i++) {
            for (int o = 0; o < edge_oi.size(); o++) {
                double outval = 0.0;
                int k = edge_oi[o][0].kernel_id;
                for (int e = 0; e < edge_oi[o].size(); e++) {
                    Edge& edge = edge_oi[o][e];
                    outval += input(edge.to, i) *
                              kernels[edge.kernel_id](edge.ky, edge.kx);
                }
                output_(o, i) = outval + bias[k];
            }
        }

        return output_;    
    }

    Matrix back_propagation(const Matrix& error, double eta = 0.1,
                            double momentum = 0.5) override {
        const int n_data = error.cols();
        const int n_input = input_size_.total() * n_featmap_in;

        Matrix prev_delta = Matrix::Zero(n_input, n_data);
        for (int d = 0; d < n_data; d++) {
            for (int i = 0; i < n_input; i++) {
                for (int e = 0; e < edge_io[i].size(); e++) {
                    Edge& edge = edge_io[i][e];
                    prev_delta(i, d) += 
                        error(edge.to, d) *
                        kernels[edge.kernel_id](edge.ky, edge.kx);
                }
            }
        }

        std::vector<Matrix> current_dkernels(n_featmap_in * n_featmap_out,
                                             Matrix::Zero(kernel_size_.rows,
                                                          kernel_size_.cols));
        std::vector<double> current_dbias(n_featmap_in * n_featmap_out, 0.0);
        for (int d = 0; d < n_data; d++) {
            for (int i = 0; i < n_input; i++) {
                for (int e = 0; e < edge_io[i].size(); e++) {
                    Edge& edge = edge_io[i][e];
                    current_dkernels[edge.kernel_id](edge.ky, edge.kx) +=
                        error(edge.to, d) * input_(i, d) / 
                        (n_data * output_size_.total());
                    current_dbias[edge.kernel_id] +=
                        error(edge.to, d) / (n_data * output_size_.total());
                }
            }
        }

        for (int k = 0; k < n_featmap_in * n_featmap_out; k++) {
            for (int i = 0; i < kernel_size_.rows; i++) {
                for (int j = 0; j < kernel_size_.cols; j++) {
                    dkernels[k](i, j) = momentum * dkernels[k](i, j) +
                                        eta * current_dkernels[k](i, j);
                    dbias[k] = momentum * dbias[k] + eta * current_dbias[k];
                    kernels[k](i, j) += dkernels[k](i, j);
                    bias[k] += dbias[k];
                }
            }
        }

        return std::move(prev_delta);    
    }
    
private:
    // Private methods
    void initialize() {
        for (int fin = 0; fin < n_featmap_in; fin++) {
            for (int fout = 0; fout < n_featmap_out; fout++) {
                if (connection_table_[fin][fout] == CONNECTED) {
                    for (int yout = 0; yout < output_size_.rows; yout++) {
                        for (int xout = 0; xout < output_size_.cols; xout++) {
                            initialize_edges(fin, fout, yout, xout);
                        }
                    }                
                }
            }
        }
    }

    void initialize_edges(int fin, int fout, int yout, int xout) {
        for (int dy = 0; dy < kernel_size_.rows; dy++) {
            for (int dx = 0; dx < kernel_size_.cols; dx++) {
                int yin = yout + dy;
                int xin = xout + dx;

                int input_index = fin * input_size_.total() +
                                  (yin * input_size_.cols + xin);
                int output_index = fout * output_size_.total() +
                                   (yout * output_size_.cols + xout);
                int kernel_index = fout * n_featmap_in + fin;
                add_edge(input_index, output_index, kernel_index, dy, dx);
            }
        }
    }

    void add_edge(int input_id, int output_id, int kernel_id, int ky, int kx) {
        edge_io[input_id].emplace_back(output_id, kernel_id, ky, kx);
        edge_oi[output_id].emplace_back(input_id, kernel_id, ky, kx);    
    }

    // Private internal classes
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

    // Private parameters
    static const int UNCONNECTED;
    static const int CONNECTED;

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
};

// Initialize static const members.
const int ConvolutionLayer::UNCONNECTED = 0;
const int ConvolutionLayer::CONNECTED = 1;

#endif  // _CONVOLUTION_LAYER_H_
