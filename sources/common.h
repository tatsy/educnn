#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <cmath>
#include <cassert>

#include <Eigen/Core>

// Declare a matrix type using Eigen
// Eigenを用いた行列タイプの宣言
using ScalarType = double;
using Matrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;

// Predefined constants
// 事前定義の定数
const double PI = 4.0 * atan(1.0);
const double EPS = 1.0e-20;
const double INFTY = 1.0e20;

struct Size {
    Size() = default;
    explicit Size(int rows_, int cols_)
        : rows(rows_)
        , cols(cols_) {
    }

    int total() const {
        return rows * cols;
    }

    int rows = 0;
    int cols = 0;
};

/**
 * The uncopyable interface class to forbid copy and assignment in the derived class.
 * 継承先クラスをコピー不可にするためのインターフェース
 */
class Uncopyable {
public:
    Uncopyable() = default;
    ~Uncopyable() = default;

    Uncopyable(const Uncopyable &) = delete;
    Uncopyable &operator=(const Uncopyable &) = delete;
};

/**
 * Calculate accuracy
 * 精度計算の関数
 */
inline double accuracy(const Matrix &m1, const Matrix &m2) {
    Matrix::Index i1, i2;
    double ret = 0.0;
    for (int i = 0; i < m1.rows(); i++) {
        m1.row(i).maxCoeff(&i1);
        m2.row(i).maxCoeff(&i2);
        ret += i1 == i2 ? 1.0 : 0.0;
    }
    return 100.0 * ret / m1.rows();
}

// -----------------------------------------------------------------------------
// Assertion with message
// -----------------------------------------------------------------------------

#ifndef __FUNCTION_NAME__
#if defined(_WIN32) || defined(__WIN32__)
#define __FUNCTION_NAME__ __FUNCTION__
#else
#define __FUNCTION_NAME__ __func__
#endif
#endif

#undef NDEBUG
#ifndef NDEBUG
#define Assertion(PREDICATE, ...)                                                                             \
    do {                                                                                                      \
        if (!(PREDICATE)) {                                                                                   \
            std::cerr << "Asssertion \"" << #PREDICATE << "\" failed in " << __FILE__ << " line " << __LINE__ \
                      << " in function \"" << (__FUNCTION_NAME__) << "\""                                     \
                      << " : ";                                                                               \
            fprintf(stderr, __VA_ARGS__);                                                                     \
            std::cerr << std::endl;                                                                           \
            std::abort();                                                                                     \
        }                                                                                                     \
    } while (false)
#else  // NDEBUG
#define Assertion(PREDICATE, MSG) \
    do {                          \
    } while (false)
#endif  // NDEBUG

#endif  // _COMMON_H_
