#ifdef _MSC_VER
#pragma once
#endif

#ifndef _LOSSES_H_
#define _LOSSES_H_

#include "common.h"

/**
 * Base class for loss functions.
 * �덷�֐��̂��߂̊��N���X
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
 * �����G���g���s�[�덷
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
 * �ΐ��\�t�g�}�b�N�X�Ŋ��������ꂽ�o�͂ɑ΂��镉�l�ΐ��ޓx�֐�
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
