#ifdef _MSC_VER
#pragma once
#endif

#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cmath>
#include <ctime>
#include <algorithm>

#include "common.h"

/**
 * Random number generator with XOR shift.
 */
class Random : private Uncopyable {
public:
    static Random &getInstance() {
        static Random instance((unsigned int)time(0));
        return instance;
    }

    ~Random() {
    }

    unsigned int nextInt() {
        const unsigned int t = seed_[0] ^ (seed_[0] << 11);
        seed_[0] = seed_[1];
        seed_[1] = seed_[2];
        seed_[2] = seed_[3];
        return seed_[3] = (seed_[3] ^ (seed_[3] >> 19)) ^ (t ^ (t >> 8));
    }

    //! Sample integer from [a, b]
    unsigned int nextInt(unsigned int a, unsigned int b) {
        unsigned int v = (unsigned int)(nextReal() * (b - a + 1) + a);
        return std::min(v, b);
    }

    //! Sample number from [0, 1]
    double nextReal() {
        return (double)nextInt() * coefficient_;
    }

    double normal() {
        const double r1 = nextReal();
        const double r2 = nextReal();
        return sqrt(-2.0 * log(r1)) * sin(2.0 * PI * r2);
    }

private:
    Random(unsigned int seed)
        : seed_{}
        , coefficient_(1.0f / UINT_MAX)
        , coefficient2_(1.0f / 16777216.0f) {
        unsigned int s = seed;
        for (int i = 1; i <= 4; i++) {
            seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
        }
    }

    unsigned int seed_[4];
    const float coefficient_;
    const float coefficient2_;
};

#endif  // _RANDOM_H_
