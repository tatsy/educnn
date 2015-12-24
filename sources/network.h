#ifdef _MSC_VER
#pragma once
#endif

#ifndef _NETWORK_H_
#define _NETWORK_H_

#include <vector>

#include "abstract_layer.h"

class Network {
private:
    std::vector<AbstractLayer*> layers_;
    std::vector<Matrix> batches_;
    std::vector<Matrix> labels_;
    int batchsize_;

public:
    Network();
    Network(const std::vector<AbstractLayer*>& layers, const Matrix& data, const Matrix& labels, int batchsize);

    virtual ~Network();

    void train(int epoch, double eta, double lambda);

    Matrix predict(const Matrix& data);
    

private:
    const Matrix& forward_propagation(const Matrix& input);
    void back_propagation(const Matrix& delta, double eta, double lambda);
    double negative_log_likelihood(const Matrix& input, const Matrix& output);
};

#endif  // _NETWORK_H_
