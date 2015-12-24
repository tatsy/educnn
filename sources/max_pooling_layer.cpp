#include "max_pooling_layer.h"

MaxPoolingLayer::MaxPoolingLayer(Random* rng, Size input_size, Size pool_size, int n_featmap)
    : AbstractLayer()
    , rng_(rng)
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

    scale.resize(n_featmap);
    bias.resize(n_featmap);
    dscale.resize(n_featmap);
    dbias.resize(n_featmap);
    for (int i = 0; i < n_featmap; i++) {
        scale[i] = 1.0;
        bias[i] = 0.0;
        dscale[i] = 0.0;
        dbias[i] = 0.0;
    }

    initialize();
}

MaxPoolingLayer::~MaxPoolingLayer() {
}

const Matrix& MaxPoolingLayer::forward_propagation(const Matrix& input) {
    int n_output = edge_oi.size();
    int n_data = input.cols();
        
    input_ = input;
    output_ = Matrix::Zero(n_output, n_data);
    for (int d = 0; d < n_data; d++) {
        for (int o = 0; o < n_output; o++) {
            double maxval = -INFTY;
            int active_index = 0;
            for (int e = 0; e < edge_oi[o].size(); e++) {
                Edge& edge = edge_oi[o][e];
                edge.active = false;
                edge_io[edge.to][edge.rev].active = false;
                if (maxval < input(edge.to, d)) {
                    maxval = input(edge.to, d);
                    active_index = e;
                }
            }


            Edge& active_e = edge_oi[o][active_index];
            output_(o, d) = scale[active_e.featmap] * input(active_e.to, d) + bias[active_e.featmap];
            active_e.active = true;
            edge_io[active_e.to][active_e.rev].active = true;
        }
    }

    return output_;
}

Matrix MaxPoolingLayer::back_propagation(const Matrix& error, double eta, double momentum) {
    const int n_data = error.cols();
    const int n_input = input_size_.total() * n_featmap_;
    Matrix prev_error = Matrix::Zero(n_input, n_data);
    for (int d = 0; d < n_data; d++) {
        for (int i = 0; i < n_input; i++) {
            for (int e = 0; e < edge_io[i].size(); e++) {
                Edge& edge = edge_io[i][e];
                if (edge.active) {
                    prev_error(i, d) += scale[edge.featmap] * error(edge.to, d);
                }
            }
        }       
    }

    std::vector<double> current_dscale(n_featmap_, 0.0);
    std::vector<double> current_dbias(n_featmap_, 0.0);
    for (int d = 0; d < n_data; d++) {
        for (int i = 0; i < n_input; i++) {
            for (int e = 0; e < edge_io[i].size(); e++) {
                Edge& edge = edge_io[i][e];
                if (edge.active) {
                    current_dscale[edge.featmap] += input_(i, d) * error(edge.to, d) / (n_data * output_size_.total());
                    current_dbias[edge.featmap] += error(edge.to, d) / (n_data * output_size_.total());
                }
            }
        }
    }

    for (int i = 0; i < n_featmap_; i++) {
        dscale[i] = momentum * dscale[i] + eta * current_dscale[i];
        dbias[i] = momentum * dbias[i] + eta * current_dbias[i];
        scale[i] += dscale[i];
        bias[i] += dbias[i];
    }

    return std::move(prev_error);
}

void MaxPoolingLayer::initialize() {
    for (int f = 0; f < n_featmap_; f++) {
        for (int yout = 0; yout < output_size_.rows; yout++) {
            for (int xout = 0; xout < output_size_.cols; xout++) {
                for (int dy = 0; dy < pool_size_.rows; dy++) {
                    for (int dx = 0; dx < pool_size_.cols; dx++) {
                        int yin = yout * pool_size_.rows + dy;
                        int xin = xout * pool_size_.cols + dx;

                        int input_index = f * input_size_.total() + (yin * input_size_.cols + xin);
                        int output_index = f * output_size_.total() + (yout * output_size_.cols + xout);
                        add_edge(input_index, output_index, f);
                    }
                }
            }
        }
    }                
}

void MaxPoolingLayer::add_edge(int in, int out, int f) {
    int n_io = edge_io[in].size();
    int n_oi = edge_oi[out].size();
    edge_io[in].emplace_back(out, f, n_oi);
    edge_oi[out].emplace_back(in, f, n_io);
}
