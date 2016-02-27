#ifdef _MSC_VER
#pragma once
#endif

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "common.h"

inline Matrix sigmoid(const Matrix& x) {
    return std::move(1.0 / (1.0 + (-x).array().exp()));
}

inline Matrix sigmoid_deriv(const Matrix& y) {
    return std::move(y.array() * (1.0 - y.array()));
}

#endif  // _ACTIVATION_H_