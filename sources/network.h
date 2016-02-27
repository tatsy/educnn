#ifdef _MSC_VER
#pragma once
#endif

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>

#include "abstract_layer.h"

class Network {
public:
    // Public methods
    Network()
        : layers_() {
    }

    Network(const std::vector<AbstractLayer*>& layers, const Matrix& data,
            const Matrix& labels, int batchsize)
        : layers_(layers)
        , batches_()
        , labels_()
        , batchsize_(batchsize) {
        const int n_batches = (data.cols() + batchsize - 1) / batchsize;
        batches_.resize(n_batches);
        labels_.resize(n_batches);
        for (int k = 0; k < n_batches; k++) {
            int batch_start = k * batchsize;
            int batch_end = std::min((int)data.cols(), (k + 1) * batchsize);

            batches_[k] = Matrix(data.rows(), batch_end - batch_start);
            for (int i = 0; i < data.rows(); i++) {
                for (int j = batch_start; j < batch_end; j++) {
                    batches_[k](i, j - batch_start) = data(i, j);
                }
            }

            labels_[k] = Matrix(labels.rows(), batch_end - batch_start);
            for (int i = 0; i < labels.rows(); i++) {
                for (int j = batch_start; j < batch_end; j++) {
                    labels_[k](i, j - batch_start) = labels(i, j);
                }
            }
        }
    }

    virtual ~Network() {}

    void train(int epoch, double eta, double lambda) {
        const int n_batches = batches_.size();
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < n_batches; j++) {
                const Matrix& output = forward_propagation(batches_[j]);
                back_propagation(labels_[j] - output, eta, lambda);
                double cost = negative_log_likelihood(labels_[j], output);
                printf("Epoch %03d-%03d: %f\n", i + 1, j + 1, cost);
            }
        }
    }

    Matrix predict(const Matrix& data) {
        return std::move(forward_propagation(data));
    }


private:
    // Private methods
    const Matrix& forward_propagation(const Matrix& input) {
        const int n_layers = layers_.size();
        for (int i = 0; i < n_layers; i++) {
            if (i == 0) {
                layers_[i]->forward_propagation(input);
            }
            else {
                layers_[i]->forward_propagation(layers_[i - 1]->output());
            }
        }
        return layers_[n_layers - 1]->output();

    }

    void back_propagation(const Matrix& delta, double eta, double lambda) {
        const int n_layers = layers_.size();

        Matrix current = delta;
        for (int i = n_layers - 1; i >= 0; i--) {
            current = layers_[i]->back_propagation(current, eta, lambda);
        }
    }

    double negative_log_likelihood(const Matrix& input, const Matrix& output) {
        Matrix pos = input.array() * output.array().log();
        Matrix neg = (1.0 - input.array()) * (1.0 - output.array()).log();
        return -(pos + neg).colwise().sum().mean();

    }

    // Private parameters
    std::vector<AbstractLayer*> layers_;
    std::vector<Matrix> batches_;
    std::vector<Matrix> labels_;
    int batchsize_;

};  // class Network

#endif  // _NETWORK_H_
