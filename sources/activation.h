#ifdef _MSC_VER
#pragma once
#endif

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "abstract_layer.h"

/**
 * ReLU activation function
 * ReLU 活性化関数
 */
class ReLU : public AbstractLayer {
public:
    // Public methods
    ReLU()
        : AbstractLayer() {
    }

    virtual ~ReLU() {
    }

    const Matrix &forward(const Matrix &input) override {
        input_ = input;
        output_ = input.cwiseMax(0.0);
        return output_;
    }

    Matrix backward(const Matrix &dLdy, double lr = 0.1, double momentum = 0.5) override {
        Matrix dLdx(input_.rows(), input_.cols());
        for (int b = 0; b < input_.rows(); b++) {
            for (int i = 0; i < input_.cols(); i++) {
                if (input_(b, i) >= 0.0) {
                    dLdx(b, i) = dLdy(b, i);
                } else {
                    dLdx(b, i) = 0.0;
                }
            }
        }
        return dLdx;
    }
};

/**
 * Sigmoid activation function
 * シグモイド活性化関数
 */
class Sigmoid : public AbstractLayer {
public:
    // Public methods
    Sigmoid()
        : AbstractLayer() {
    }

    virtual ~Sigmoid() {
    }

    const Matrix &forward(const Matrix &input) override {
        input_ = input;
        output_ = 1.0 / (1.0 + (-input).array().exp());
        return output_;
    }

    Matrix backward(const Matrix &dLdy, double lr = 0.1, double momentum = 0.5) override {
        const Matrix dydx = output_.array() * (1.0 - output_.array());
        return dLdy.cwiseProduct(dydx);
    }
};

/**
 * Softmax activation function (row-wise operation)
 * ソフトマックス活性化関数 (行ごとに適用される)
 */
class Softmax : public AbstractLayer {
public:
    // Public methods
    Softmax() {
    }

    virtual ~Softmax() {
    }

    const Matrix &forward(const Matrix &input) override {
        const int dims = (int)input.cols();
        input_ = input;

        // To increase numerical precision, inputs are divided by their max value
        // 数値計算の精度を向上させるために、入力の値をその最大値で予め割り算しておく
        const Matrix cwiseMax = input.rowwise().maxCoeff().replicate(1, dims);
        const Matrix normalized = input.cwiseQuotient(cwiseMax);
        const Matrix numer = normalized.array().exp().matrix();
        const Matrix denom = numer.rowwise().sum().replicate(1, dims);
        output_ = numer.cwiseQuotient(denom);
        return output_;
    }

    Matrix backward(const Matrix &dLdy, double lr = 0.1, double momentum = 0.5) override {
        const int batchsize = (int)dLdy.rows();
        const int dims = (int)dLdy.cols();

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
 * Log-softmax activation function (row-wise operation)
 * 対数ソフトマックス活性化関数 (行ごとに適用される)
 */
class LogSoftmax : public AbstractLayer {
public:
    // Public methods
    LogSoftmax() {
    }

    virtual ~LogSoftmax() {
    }

    const Matrix &forward(const Matrix &input) override {
        const int dims = (int)input.cols();
        input_ = input;

        // To avoid loss of trailing digits, inputs are subtracted by their max value
        // 情報落ち誤差を防ぐために、入力の値をその最大値で予め引き算しておく
        const Matrix cwiseMax = input.rowwise().maxCoeff().replicate(1, dims);
        const Matrix diff = input - cwiseMax;
        const Matrix sumexpDiff = diff.array().exp().matrix().rowwise().sum();
        const Matrix logsumexp = sumexpDiff.array().log().matrix().replicate(1, dims) + cwiseMax;
        output_ = input - logsumexp;
        return output_;
    }

    Matrix backward(const Matrix &dLdy, double lr = 0.1, double momentum = 0.5) override {
        const int batchsize = (int)dLdy.rows();
        const int dims = (int)dLdy.cols();

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
