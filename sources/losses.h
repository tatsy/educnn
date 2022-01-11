#ifdef _MSC_VER
#pragma once
#endif

#ifndef _LOSSES_H_
#define _LOSSES_H_

#include "common.h"

/**
 * Base class for loss functions.
 * 誤差関数のための基底クラス
 */
class AbstractLoss : private Uncopyable {
public:
    AbstractLoss() = default;
    virtual ~AbstractLoss() = default;

    virtual const Matrix &forward(const Matrix &pred, const Matrix &real) = 0;
    virtual Matrix backward() = 0;

protected:
    Matrix input_ = {};
    Matrix target_ = {};
    Matrix output_ = {};
};

/**
 * Cross-entropy loss
 * 交差エントロピー誤差
 */
class CrossEntropyLoss : public AbstractLoss {
public:
    CrossEntropyLoss() = default;
    virtual ~CrossEntropyLoss() = default;

    const Matrix &forward(const Matrix &input, const Matrix &target) override {
        const Matrix prod = -target.cwiseProduct(input.array().log().matrix());
        input_ = input;
        target_ = target;
        output_ = prod.rowwise().sum();
        return output_;
    }

    Matrix backward() override {
        return -target_.cwiseQuotient(input_);
    }
};

/**
 * Negative log likelihood for log-softmax activation
 * 対数ソフトマックスで活性化された出力に対する負値対数尤度関数
 */
class NLLLoss : public AbstractLoss {
public:
    NLLLoss() = default;
    virtual ~NLLLoss() = default;

    const Matrix &forward(const Matrix &input, const Matrix &target) override {
        const Matrix prod = -target.cwiseProduct(input);
        input_ = input;
        target_ = target;
        output_ = prod.rowwise().sum();
        return output_;
    }

    Matrix backward() override {
        return -target_;
    }
};

#endif  // _LOSSES_H_
