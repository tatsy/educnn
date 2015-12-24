#ifdef _MSC_VER
#pragma once
#endif

#ifndef _FULLY_CONNECTED_LAYER_H_
#define _FULLY_CONNECTED_LAYER_H_

#include "random.h"
#include "abstract_layer.h"

class FullyConnectedLayer : public AbstractLayer {
private:
    Random* rng_ = nullptr;
    int input_size_ = 0;
    int output_size_ = 0;

    Matrix W = {};
    Matrix b = {};
    Matrix dW = {};
    Matrix db = {};

public:
    FullyConnectedLayer();
    FullyConnectedLayer(Random* rng, int input_size, int output_size);

    virtual ~FullyConnectedLayer();

    const Matrix& forward_propagation(const Matrix& input) override;

    Matrix back_propagation(const Matrix& err, double eta = 0.1, double momentum = 0.5) override;
};

#endif  // _FULLY_CONNECTED_LAYER_H_
