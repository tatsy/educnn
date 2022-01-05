#ifdef _MSC_VER
#pragma once
#endif

#ifndef _CONVOLUTION_LAYER_H_
#define _CONVOLUTION_LAYER_H_

#include <vector>

#include "openmp.h"
#include "random.h"
#include "abstract_layer.h"

class ConvolutionLayer : public AbstractLayer {
public:
    // Public methods
    ConvolutionLayer(Size input_size, Size kernel_size, int in_channels, int out_channels)
        : AbstractLayer()
        , input_size_(input_size)
        , kernel_size_(kernel_size)
        , output_size_()
        , in_channels(in_channels)
        , out_channels(out_channels) {
        output_size_.rows = input_size_.rows - kernel_size_.rows + 1;
        output_size_.cols = input_size_.cols - kernel_size_.cols + 1;

        W.resize(in_channels * out_channels);
        b.resize(out_channels);
        dW.resize(in_channels * out_channels);
        db.resize(out_channels);

        Random &rng = Random::getInstance();
        for (int k = 0; k < in_channels * out_channels; k++) {
            W[k] = Matrix(kernel_size_.rows, kernel_size_.cols);
            dW[k] = Matrix(kernel_size.rows, kernel_size_.cols);
            for (int i = 0; i < kernel_size_.rows; i++) {
                for (int j = 0; j < kernel_size_.cols; j++) {
                    W[k](i, j) = rng.normal() * 0.1;
                    dW[k](i, j) = 0.0;
                }
            }
        }

        for (int k = 0; k < out_channels; k++) {
            b[k] = 0.0;
            db[k] = 0.0;
        }

        // Initialize bipartite graph between input and output
        edges_o2i.resize(output_size_.total() * out_channels);
        initialize();
    }

    virtual ~ConvolutionLayer() {
    }

    const Matrix &forward(const Matrix &input) override {
        const int batchsize = (int)input.rows();
        const int n_output = output_size_.total() * out_channels;

        input_ = input;
        output_ = Matrix::Zero(batchsize, n_output);
        for (int b = 0; b < batchsize; b++) {
            omp_parallel_for(int o = 0; o < n_output; o++) {
                double accum = 0.0;
                for (int e = 0; e < edges_o2i[o].size(); e++) {
                    Edge &edge = edges_o2i[o][e];
                    accum += input(b, edge.to) * W[edge.kernel_id](edge.ky, edge.kx);
                }
                const int out_ch = o / output_size_.total();
                output_(b, o) = accum + this->b[out_ch];
            }
        }

        return output_;
    }

    Matrix backward(const Matrix &dLdy, double eta = 0.1, double momentum = 0.5) override {
        const int batchsize = (int)dLdy.rows();
        const int n_input = input_size_.total() * in_channels;
        const int n_output = output_size_.total() * out_channels;

        Matrix dLdx = Matrix::Zero(batchsize, n_input);
        for (int b = 0; b < batchsize; b++) {
            omp_parallel_for(int o = 0; o < n_output; o++) {
                for (int e = 0; e < edges_o2i[o].size(); e++) {
                    Edge &edge = edges_o2i[o][e];

#pragma omp atomic
                    dLdx(b, edge.to) += dLdy(b, o) * W[edge.kernel_id](edge.ky, edge.kx);
                }
            }
        }

        std::vector<Matrix> current_dW(in_channels * out_channels, Matrix::Zero(kernel_size_.rows, kernel_size_.cols));
        std::vector<double> current_db(out_channels, 0.0);
        for (int b = 0; b < batchsize; b++) {
            omp_parallel_for(int o = 0; o < n_output; o++) {
                for (int e = 0; e < edges_o2i[o].size(); e++) {
                    Edge &edge = edges_o2i[o][e];

#pragma omp atomic
                    current_dW[edge.kernel_id](edge.ky, edge.kx) +=
                        dLdy(b, o) * input_(b, edge.to) / (batchsize * output_size_.total());
                }
                const int out_ch = o / output_size_.total();

#pragma omp atomic
                current_db[out_ch] += dLdy(b, o) / (batchsize * output_size_.total());
            }
        }

        omp_parallel_for(int k = 0; k < in_channels * out_channels; k++) {
            for (int i = 0; i < kernel_size_.rows; i++) {
                for (int j = 0; j < kernel_size_.cols; j++) {
                    dW[k](i, j) = momentum * dW[k](i, j) + eta * current_dW[k](i, j);
                    W[k](i, j) -= dW[k](i, j);
                }
            }
        }

        omp_parallel_for(int k = 0; k < out_channels; k++) {
            b[k] -= db[k];
            db[k] = momentum * db[k] + eta * current_db[k];
        }

        return dLdx;
    }

private:
    // Private methods
    void initialize() {
        for (int fin = 0; fin < in_channels; fin++) {
            for (int fout = 0; fout < out_channels; fout++) {
                for (int yout = 0; yout < output_size_.rows; yout++) {
                    for (int xout = 0; xout < output_size_.cols; xout++) {
                        initialize_edges(fin, fout, yout, xout);
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

                int input_index = fin * input_size_.total() + (yin * input_size_.cols + xin);
                int output_index = fout * output_size_.total() + (yout * output_size_.cols + xout);
                int kernel_index = fout * in_channels + fin;
                add_edge(input_index, output_index, kernel_index, dy, dx);
            }
        }
    }

    void add_edge(int input_id, int output_id, int kernel_id, int ky, int kx) {
        edges_o2i[output_id].emplace_back(input_id, kernel_id, ky, kx);
    }

    // Private internal classes
    struct Edge {
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

    Size input_size_ = {};
    Size kernel_size_ = {};
    Size output_size_ = {};
    int in_channels = 0;
    int out_channels = 0;

    std::vector<std::vector<Edge>> edges_o2i = {};

    std::vector<Matrix> W = {};
    std::vector<double> b = {};
    std::vector<Matrix> dW = {};
    std::vector<double> db = {};
};

#endif  // _CONVOLUTION_LAYER_H_
