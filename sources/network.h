#ifdef _MSC_VER
#pragma once
#endif

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>

#include "random.h"
#include "losses.h"
#include "abstract_layer.h"

class Network {
public:
    // Public methods
    Network()
        : layers_() {
    }

    Network(const std::vector<AbstractLayer*>& layers, AbstractLoss *criterion)
        : layers_(layers)
        , criterion_(criterion) {
    }

    virtual ~Network() {}

    void train(const Matrix &data, const Matrix &labels, int epoch, int batchsize, double eta, double lambda) {
        // Shuffle data indices
        Random &rng = Random::getInstance();
        const int n_data = data.rows();

        for (int e = 0; e < epoch; e++) {
            // Shuffle data indices
            std::vector<int> indices(n_data);
            for (int i = 0; i < n_data; i++) {
                indices[i] = i;
            }

            for (int i = 0; i < n_data; i++) {
                const int k = rng.nextInt(i, n_data - 1);
                std::swap(indices[i], indices[k]);
            }

            // Train with each batch
            const int n_batches = (n_data + batchsize - 1) / batchsize;
            for (int j = 0; j < n_batches; j++) {
                // Construct a batch of data
                const int B = std::min(batchsize, n_data - j * batchsize);
                Matrix batch_data(B, data.cols());
                Matrix batch_labels(B, labels.cols());
                for (int b = 0; b < B; b++) {
                    batch_data.row(b) = data.row(indices[j * batchsize + b]);
                    batch_labels.row(b) = labels.row(indices[j * batchsize + b]);
                }

                // Process
                const Matrix& output = forward_propagation(batch_data);
                const Matrix& losses = criterion_->forward_propagation(output, batch_labels);
                
                back_propagation(criterion_->back_propagation(), eta, lambda);
                printf("Epoch %03d-%03d: %f\n", e + 1, j + 1, losses.mean());
            }
        }
    }

    Matrix predict(const Matrix& data) {
        return forward_propagation(data);
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
        Matrix prod = -1.0 * input.array() * output.array();
        return prod.colwise().sum().mean();

    }

    // Private parameters
    std::vector<AbstractLayer*> layers_;
    AbstractLoss* criterion_;

};  // class Network

#endif  // _NETWORK_H_
