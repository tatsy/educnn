#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MNIST_H_
#define _MNIST_H_

#include <iostream>
#include <fstream>

#include "common.h"
#include "directories.h"

namespace mnist {

// -----------------------------------------------------------------------------
// MNIST utility function definitions
// -----------------------------------------------------------------------------

namespace {

int big_endian(unsigned char* b) {
    int ret = 0;
    for (int i = 0; i < 4; i++) {
        ret = (ret << 8) | b[i];
    }
    return ret;
}

Matrix load_data(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open train data!!" << std::endl;
        exit(1);
    }

    unsigned char b[4];
    ifs.read((char*)b, sizeof(char) * 4);

    ifs.read((char*)b, sizeof(char) * 4);
    int nimg = big_endian(b);

    ifs.read((char*)b, sizeof(char) * 4);
    int rows = big_endian(b);

    ifs.read((char*)b, sizeof(char) * 4);
    int cols = big_endian(b);

    unsigned char* buf = new unsigned char[rows * cols];
    Matrix ret(rows * cols, nimg);
    for (int j = 0; j < nimg; j++) {
        ifs.read((char*)buf, sizeof(char) * rows * cols);
        for (int i = 0; i < rows * cols; i++) {
            ret(i, j) = buf[i] / 255.0;
        }
    }
    delete[] buf;

    ifs.close();

    return std::move(ret);
}

Matrix load_label(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);

    unsigned char b[4];
    ifs.read((char*)b, sizeof(char) * 4);

    ifs.read((char*)b, sizeof(char) * 4);
    int nimg = big_endian(b);

    Matrix ret(10, nimg);
    for (int i = 0; i < nimg; i++) {
        char digit;
        ifs.read((char*)&digit, sizeof(char));
        ret(digit, i) = 1.0;
    }

    ifs.close();

    return std::move(ret);
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// MNIST parser definitions
// -----------------------------------------------------------------------------

Matrix train_data() {
    return std::move(load_data(train_image_file));
}

Matrix train_label() {
    return std::move(load_label(train_label_file));
}

Matrix test_data() {
    return std::move(load_data(test_image_file));
}

Matrix test_label() {
    return std::move(load_label(test_label_file));
}

}  // namespace mnist

#endif  // _MNIST_H_
