#include "activation.h"

Matrix sigmoid(const Matrix& x) {
    return std::move(1.0 / (1.0 + (-x).array().exp()));
}

Matrix sigmoid_deriv(const Matrix& y) {
    return std::move(y.array() * (1.0 - y.array()));
}
