#ifdef _MSC_VER
#pragma once
#endif

#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include "common.h"

Matrix sigmoid(const Matrix& x);

Matrix sigmoid_deriv(const Matrix& y);

#endif  // _ACTIVATION_H_