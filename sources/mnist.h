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
    std::ifstream reader(filename.c_str(), std::ios::in | std::ios::binary);
    if (reader.fail()) {
        std::cerr << "Failed to open data: " << filename << std::endl;
        exit(1);
    }

    unsigned char b[4];
    reader.read((char*)b, sizeof(char) * 4);

    reader.read((char*)b, sizeof(char) * 4);
    const int n_image = big_endian(b);

    reader.read((char*)b, sizeof(char) * 4);
    const int rows = big_endian(b);

    reader.read((char*)b, sizeof(char) * 4);
    const int cols = big_endian(b);

    uint8_t* buf = new uint8_t[rows * cols];
    Matrix ret(n_image, rows * cols);
    for (int i = 0; i < n_image; i++) {
        reader.read((char*)buf, sizeof(char) * rows * cols);
        for (int j = 0; j < rows * cols; j++) {
            ret(i, j) = buf[j] / 255.0;
        }
    }
    delete[] buf;

    reader.close();

    return ret;
}

Matrix load_label(const std::string& filename) {
    std::ifstream reader(filename.c_str(), std::ios::in | std::ios::binary);
    if (reader.fail()) {
        std::cerr << "Failed to open labels: " << filename << std::endl;
        exit(1);
    }

    unsigned char b[4];
    reader.read((char*)b, sizeof(char) * 4);

    reader.read((char*)b, sizeof(char) * 4);
    const int n_image = big_endian(b);

    Matrix ret(n_image, 10);
    ret.setZero();
    for (int i = 0; i < n_image; i++) {
        char digit;
        reader.read((char*)&digit, sizeof(char));
        ret(i, digit) = 1.0;
    }

    reader.close();

    return ret;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// MNIST parser definitions
// -----------------------------------------------------------------------------

Matrix train_data() {
    return load_data(train_image_file);
}

Matrix train_labels() {
    return load_label(train_label_file);
}

Matrix test_data() {
    return load_data(test_image_file);
}

Matrix test_labels() {
    return load_label(test_label_file);
}

}  // namespace mnist

#endif  // _MNIST_H_
