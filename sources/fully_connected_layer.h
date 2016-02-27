#ifdef _MSC_VER
#pragma once
#endif

#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

#include "random.h"
#include "activation.h"
#include "abstract_layer.h"

class FullyConnectedLayer : public AbstractLayer {
public:
    // Public methods
    FullyConnectedLayer()
        : AbstractLayer() {
    }

    FullyConnectedLayer(Random* rng, int input_size, int output_size)
        : AbstractLayer()
        , rng_(rng)
        , input_size_(input_size)
        , output_size_(output_size)
        , W(output_size, input_size)
        , b(output_size, 1)
        , dW(output_size, input_size)
        , db(output_size, 1) {
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W(i, j) = rng->normal() * 0.1;
                dW(i, j) = 0.0;
            }
            b(i, 0) = 0.0;
            db(i, 0) = 0.0;
        }
    }


    virtual ~FullyConnectedLayer() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        const int n_data = input.cols();
        input_ = input;
        output_ = sigmoid(W * input + b.replicate(1, n_data));
        return output_;
    }

    Matrix back_propagation(const Matrix& err, double eta = 0.1,
                            double momentum = 0.5) override {
        const int n_data = err.cols();
        Matrix delta = err.cwiseProduct(sigmoid_deriv(output_));
        Matrix prev_err = W.transpose() * delta;

        Matrix current_dW = delta * input_.transpose() / n_data;
        Matrix current_db = delta.rowwise().mean();
        dW = momentum * dW + eta * current_dW;
        db = momentum * db + eta * current_db;

        W += dW;
        b += db;

        return std::move(prev_err);
    }

private:
    // Private parameters
    Random* rng_ = nullptr;
    int input_size_ = 0;
    int output_size_ = 0;

    Matrix W = {};
    Matrix b = {};
    Matrix dW = {};
    Matrix db = {};

};  // class FullyConnectedLayer

#endif  // _FULLY_CONNECTED_LAYER_H_
