#ifdef _MSC_VER
#pragma once
#endif

#ifndef _LOSSES_H_
#define _LOSSES_H_

/**
 * @brief: Base class for loss functions.
 */
class AbstractLoss : private Uncopyable {
public:
    AbstractLoss() {}
    virtual ~AbstractLoss() {}

    virtual const Matrix& forward_propagation(const Matrix& pred, const Matrix &real) = 0;
    virtual Matrix back_propagation() = 0;

protected:
    Matrix input_ = {};
    Matrix target_ = {};
    Matrix output_ = {};
};

/**
 * Cross-entropy loss
 */
class CrossEntropyLoss : public AbstractLoss {
public:
    CrossEntropyLoss() {}
    virtual ~CrossEntropyLoss() {}

    const Matrix& forward_propagation(const Matrix &input, const Matrix &target) override {
        const Matrix prod = -target.cwiseProduct(input.array().log().matrix());
        input_ = input;
        target_ = target;
        output_ = prod.rowwise().sum();
        return output_;
    }

    Matrix back_propagation() override {
        return -target_.cwiseQuotient(input_);
    }
};

/**
 * Negative log likelihood for log-softmax activation
 */
class NLLLoss : public AbstractLoss {
public:
    NLLLoss() {}
    virtual ~NLLLoss() {}

    const Matrix& forward_propagation(const Matrix &input, const Matrix &target) override {
        const Matrix prod = -target.cwiseProduct(input);
        input_ = input;
        target_ = target;
        output_ = prod.rowwise().sum();
        return output_;
    }

    Matrix back_propagation() override {
        return -target_;
    }
};

#endif  // _LOSSES_H_