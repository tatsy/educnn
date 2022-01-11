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
        // X. Glorot's standard deviation for parameter initialization
        // X. Glorotによるパラメータ初期化のための標準偏差
        const double xg_stddev = sqrt(2.0 / (input_size_ + output_size_));

        // Parameter initialization
        // パラメータの初期化
        Random &rng = Random::getInstance();
        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                W(i, j) = rng.normal() * xg_stddev;
                dW(i, j) = 0.0;
            }
            b(0, i) = 0.0;
            db(0, i) = 0.0;
        }
    }

    virtual ~FullyConnectedLayer() {
    }

    const Matrix &forward(const Matrix &input) override {
        // Simple linear operation (y = Wx + b)
        // 単純な線形演算 (y = Wx + b)
        const int batchsize = (int)input.rows();
        input_ = input;
        output_ = input * W.transpose() + b.replicate(batchsize, 1);
        return output_;
    }

    Matrix backward(const Matrix &dLdy, double lr = 0.1, double momentum = 0.5) override {
        // Assum x and y are input and output of this layer, hence back-prop transforms dLdy to dLdx.
        // xとyがこのレイヤーの入出力だと仮定. 誤差逆伝播のためにdLdyをdLdxに変換する
        const int batchsize = (int)dLdy.rows();
        const Matrix dLdx = dLdy * W;

        // Momentum SGD
        // 慣性つき確率的最急降下法
        const Matrix current_dW = dLdy.transpose() * input_;
        const Matrix current_db = dLdy.colwise().sum();
        dW = momentum * dW + lr * current_dW;
        db = momentum * db + lr * current_db;

        // Update parameters
        // パラメータの更新
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
