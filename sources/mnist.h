#ifdef _MSC_VER
#pragma once
#endif

#ifndef _MNIST_H_
#define _MNIST_H_

#include <iostream>
#include <fstream>

#include "common.h"
#include "directories.h"

// -----------------------------------------------------------------------------
// MNIST utility function definitions
// -----------------------------------------------------------------------------

namespace {

/**
 * Convert integer from little endian to big endian
 * リトルエンディアンとビッグエンディアンの相互変換
 */
inline uint32_t change_endian(uint32_t x) {
    uint32_t ret = 0;
    for (int i = 0; i < 4; i++) {
        ret = (ret << 8) | (x & 0xff);
        x >>= 8;
    }
    return ret;
}

/**
 * Load image data
 * 画像データの読み込み
 */
inline Matrix load_images(const std::string &filename) {
    std::ifstream reader(filename.c_str(), std::ios::in | std::ios::binary);
    if (reader.fail()) {
        std::cerr << "Failed to open data: " << filename << std::endl;
        exit(1);
    }

    uint32_t temp;

    // read magic number
    // マジックナンバーの読み込み
    reader.read((char *)&temp, sizeof(char) * 4);
    const int magic = change_endian(temp);
    Assertion(magic == 2051, "Invalid magic number!");

    // read number of data
    // データの数の読み込み
    reader.read((char *)&temp, sizeof(char) * 4);
    const int n_image = change_endian(temp);

    // read image height (# of rows)
    // 画像の高さ(行数)を読む
    reader.read((char *)&temp, sizeof(char) * 4);
    const int rows = change_endian(temp);

    // read image width (# of columns)
    // 画像の幅(列数)を読む
    reader.read((char *)&temp, sizeof(char) * 4);
    const int cols = change_endian(temp);

    // read pixel values
    // 画像の画素値を読む
    uint8_t *buf = new uint8_t[rows * cols];
    Matrix ret = Matrix::Zero(n_image, rows * cols);
    for (int i = 0; i < n_image; i++) {
        reader.read((char *)buf, sizeof(char) * rows * cols);
        for (int j = 0; j < rows * cols; j++) {
            ret(i, j) = buf[j] / 255.0;
        }
    }
    delete[] buf;

    reader.close();

    return ret;
}

/**
 * Load label data
 * ラベルデータの読み込み
 */
inline Matrix load_labels(const std::string &filename) {
    std::ifstream reader(filename.c_str(), std::ios::in | std::ios::binary);
    if (reader.fail()) {
        std::cerr << "Failed to open labels: " << filename << std::endl;
        exit(1);
    }

    uint32_t temp;

    // read magic number
    // マジックナンバーの読み込み
    reader.read((char *)&temp, sizeof(char) * 4);
    const int magic = change_endian(temp);
    Assertion(magic == 2049, "Invalid magic number!");

    // read number of labes
    // ラベル数の読み込み
    reader.read((char *)&temp, sizeof(char) * 4);
    const int n_image = change_endian(temp);

    // read label index and convert it to one-hot vector
    // ラベル番号を読み取ってone-hotベクトルに変換する
    Matrix ret = Matrix::Zero(n_image, 10);
    for (int i = 0; i < n_image; i++) {
        char digit;
        reader.read((char *)&digit, sizeof(char));
        ret(i, digit) = 1.0;
    }

    reader.close();

    return ret;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// MNIST parser definitions
// -----------------------------------------------------------------------------

namespace mnist {

/**
 * Load train images
 * 訓練画像の読み込み
 */
inline Matrix train_images() {
    return load_images(train_image_file);
}

/**
 * Load train labels
 * 訓練ラベルの読み込み
 */
inline Matrix train_labels() {
    return load_labels(train_label_file);
}

/**
 * Load test images
 * テスト用画像の読み込み
 */
inline Matrix test_images() {
    return load_images(test_image_file);
}

/**
 * Load test labels
 * テスト用ラベルの読み込み
 */
inline Matrix test_labels() {
    return load_labels(test_label_file);
}

}  // namespace mnist

#endif  // _MNIST_H_
