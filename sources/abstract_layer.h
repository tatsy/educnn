#ifdef _MSC_VER
#pragma once
#endif

#ifndef _ABSTRACT_LAYER_H_
#define _ABSTRACT_LAYER_H_

#include "common.h"

/**
 * @brief: Base class for neural network layers.
 */
class AbstractLayer : private Uncopyable {
protected:
    Matrix input_ = {};
    Matrix output_ = {};

public:
    AbstractLayer() {}
    virtual ~AbstractLayer() {}

    virtual const Matrix& forward_propagation(const Matrix& input) = 0;
    virtual Matrix back_propagation(const Matrix& error, double eta = 0.1, double momentum = 0.5) = 0;

    inline const Matrix& input() const { return input_; }
    inline const Matrix& output() const { return output_; }
};

#endif  // _ABSTRACT_LAYER_H_
