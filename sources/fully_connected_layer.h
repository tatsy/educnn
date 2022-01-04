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

    FullyConnectedLayer(int input_size, int output_size)
        : AbstractLayer()
        , input_size_(input_size)
        , output_size_(output_size)
        , W(output_size, input_size)
        , b(1, output_size)
        , dW(output_size, input_size)
        , db(1, output_size) {

        Random &rng = Random::getInstance();
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W(i, j) = rng.normal() * 0.1;
                dW(i, j) = 0.0;
            }
            b(0, i) = 0.0;
            db(0, i) = 0.0;
        }
    }

    virtual ~FullyConnectedLayer() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        const int batchsize = input.rows();
        input_ = input;
        output_ = input * W.transpose() + b.replicate(batchsize, 1);
        return output_;
    }

    Matrix back_propagation(const Matrix& dLdy, double eta = 0.1,
                            double momentum = 0.5) override {
        const int batchsize = dLdy.rows();
        const Matrix dLdx = dLdy * W;

        const Matrix current_dW = dLdy.transpose() * input_ / batchsize;
        const Matrix current_db = dLdy.colwise().mean();
        dW = momentum * dW + eta * current_dW;
        db = momentum * db + eta * current_db;

        W -= dW;
        b -= db;

        return dLdx;
    }

private:
    // Private parameters
    int input_size_ = 0;
    int output_size_ = 0;

    Matrix W = {};
    Matrix b = {};
    Matrix dW = {};
    Matrix db = {};

};  // class FullyConnectedLayer

#endif  // _FULLY_CONNECTED_LAYER_H_
