#ifdef _MSC_VER
#pragma once
#endif

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>

#include "progress.h"
#include "random.h"
#include "losses.h"
#include "abstract_layer.h"

class Network {
public:
    // Public methods
    Network()
        : layers_() {
    }

    Network(const std::vector<std::shared_ptr<AbstractLayer>> &layers)
        : layers_(layers) {
    }

    virtual ~Network() {
    }

    const Matrix &forward(const Matrix &input) {
        const int n_layers = (int)layers_.size();
        for (int i = 0; i < n_layers; i++) {
            if (i == 0) {
                layers_[i]->forward(input);
            } else {
                layers_[i]->forward(layers_[i - 1]->output());
            }
        }
        return layers_[n_layers - 1]->output();
    }

    void backward(const Matrix &delta, double eta, double lambda) {
        const int n_layers = (int)layers_.size();

        Matrix current = delta;
        for (int i = n_layers - 1; i >= 0; i--) {
            current = layers_[i]->backward(current, eta, lambda);
        }
    }

private:
    // Private parameters
    std::vector<std::shared_ptr<AbstractLayer>> layers_;

};  // class Network

#endif  // _NETWORK_H_
