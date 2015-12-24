#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MNIST_H_
#define _MNIST_H_

#include <iostream>
#include <fstream>

#include "common.h"

namespace mnist {

    Matrix train_data();

    Matrix train_label();

    Matrix test_data();

    Matrix test_label();

}  // namespace mnist

#endif  // _MNIST_H_
