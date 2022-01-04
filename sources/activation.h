#ifdef _MSC_VER
#pragma once
#endif

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "abstract_layer.h"

/**
 * @brief: ReLU activation function
 */
class ReLU : public AbstractLayer {
public:
     // Public methods
     ReLU()
         : AbstractLayer() {
     }

     virtual ~ReLU() {}

     const Matrix& forward_propagation(const Matrix& input) override {
         input_ = input;
         output_ = input.cwiseMax(0.0);
         return output_;
     }

     Matrix back_propagation(const Matrix &dLdy, double eta = 0.1, double momentum = 0.5) override {
         Matrix dLdx(input_.rows(), input_.cols());
         dLdx.setZero();
         for (int b = 0; b < input_.rows(); b++) {
            for (int i = 0; i < input_.cols(); i++) {
                if (input_(b, i) >= 0.0) {
                    dLdx(b, i) = dLdy(b, i);
                }
            }
         }
         return dLdx;
     }
};

/**
 * @brief: Sigmoid activation function
 */
class Sigmoid : public AbstractLayer {
public:
     // Public methods
     Sigmoid()
         : AbstractLayer() {
     }

     virtual ~Sigmoid() {}

     const Matrix& forward_propagation(const Matrix& input) override {
         input_ = input;
         output_ = 1.0 / (1.0 + (-input).array().exp());
         return output_;
     }

     Matrix back_propagation(const Matrix &dLdy, double eta = 0.1, double momentum = 0.5) override {
         const Matrix dydx = output_.array() * (1.0 - output_.array());
         return dLdy.cwiseProduct(dydx);
     }
};

/**
 * @brief: Softmax activation function (for last dimension)
 */
class Softmax : public AbstractLayer {
public:
    // Public methods
    Softmax() {
    }

    virtual ~Softmax() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        const int dims = input.cols();
        input_ = input;
        const Matrix cwiseMax = input.rowwise().maxCoeff().replicate(1, dims);
        const Matrix normalized = input.cwiseQuotient(cwiseMax);
        const Matrix numer = normalized.array().exp().matrix();
        const Matrix denom = numer.rowwise().sum().replicate(1, dims);
        output_ = numer.cwiseQuotient(denom);
        return output_;
    }

    Matrix back_propagation(const Matrix& dLdy, double eta = 0.1, double momentum = 0.5) override {
        const int batchsize = dLdy.rows();
        const int dims = dLdy.cols();

        Matrix dLdx(batchsize, dims);
        for (int b = 0; b < batchsize; b++) {
            for (int i = 0; i < dims; i++) {
                dLdx(b, i) = 0.0;
                for (int j = 0; j < dims; j++) {
                    double m;
                    if (i == j) {
                        m = output_(b, i) * (1.0 - output_(b, j));
                    } else {
                        m = -output_(b, i) * output_(b, j);
                    }
                    dLdx(b, i) += dLdy(b, j) * m;                    
                }
            }
        }

        return dLdx;
    }
};

/**
 * @brief: Log-softmax activation function (for last dimension)
 */
class LogSoftmax : public AbstractLayer {
public:
    // Public methods
    LogSoftmax() {
    }

    virtual ~LogSoftmax() {}

    const Matrix& forward_propagation(const Matrix& input) override {
        const int dims = input.cols();
        input_ = input;
        const Matrix cwiseMax = input.rowwise().maxCoeff().replicate(1, dims);
        const Matrix diff = input - cwiseMax;
        const Matrix sumexpDiff = diff.array().exp().matrix().rowwise().sum();
        const Matrix logsumexp = sumexpDiff.array().log().matrix().replicate(1, dims) + cwiseMax;
        output_ = input - logsumexp;
        return output_;
    }

    Matrix back_propagation(const Matrix& dLdy, double eta = 0.1, double momentum = 0.5) override {
        const int batchsize = dLdy.rows();
        const int dims = dLdy.cols();

        Matrix dLdx(batchsize, dims);
        for (int b = 0; b < batchsize; b++) {
            for (int i = 0; i < dims; i++) {
                dLdx(b, i) = 0.0;
                for (int j = 0; j < dims; j++) {
                    double m;
                    if (i == j) {
                        m = 1.0 - exp(output_(b, i));
                    } else {
                        m = -exp(output_(b, i));
                    }
                    dLdx(b, i) += dLdy(b, j) * m;                    
                }
            }
        }

        return dLdx;
    }
};

#endif  // _ACTIVATION_H_