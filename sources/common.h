#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <cmath>
#include <cassert>
#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;

const double PI = 4.0 * atan(1.0);
const double EPS = 1.0e-20;
const double INFTY = 1.0e20;

struct Size {
    Size() {
    }
    
    Size(int rows_, int cols_)
        : rows(rows_)
        , cols(cols_) {
    }

    inline int total() const {
        return rows * cols;
    }

    int rows = 0;
    int cols = 0;
};

/**
 * The uncopyable interface class to forbid copy and assignement
 * in the derived class.
 */
class Uncopyable {
public:
    Uncopyable() {}
    ~Uncopyable() {}

    Uncopyable(const Uncopyable&) = delete;
    Uncopyable& operator=(const Uncopyable&) = delete;
};

/**
 * Check function
 */
double check(const Matrix& m1, const Matrix& m2) {
    Matrix::Index i1, i2;
    double ret = 0.0;
    for (int j = 0; j < m1.cols(); j++) {
        m1.col(j).maxCoeff(&i1);
        m2.col(j).maxCoeff(&i2);
        ret += i1 == i2 ? 1.0 : 0.0;
    }
    return 100.0 * ret / m1.cols();
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
#define Assertion(PREDICATE, ...) \
do { \
    if (!(PREDICATE)) { \
        std::cerr << "Asssertion \"" \
        << #PREDICATE << "\" failed in " << __FILE__ \
        << " line " << __LINE__ \
        << " in function \"" << (__FUNCTION_NAME__) << "\"" \
        << " : "; \
        fprintf(stderr, __VA_ARGS__); \
        std::cerr << std::endl; \
        std::abort(); \
    } \
} while (false)
#else  // NDEBUG
#define Assertion(PREDICATE, MSG) do {} while (false)
#endif  // NDEBUG

#endif  // _COMMON_H_
