#ifdef _MSC_VER
#pragma once
#endif

#ifndef _RANDOM_H_
#define _RANDOM_H_

#include <cmath>

#include "common.h"

/** Random number generator with XOR shift.
 */
class Random {
private:
    unsigned int seed_[4];
    const float coefficient_;
    const float coefficient2_;

public:

    Random(const unsigned int initial_seed) :
        coefficient_(1.0f / UINT_MAX), coefficient2_(1.0f / 16777216.0f) {
        unsigned int s = initial_seed;
        for (int i = 1; i <= 4; i++) {
            seed_[i - 1] = s = 1812433253U * (s ^ (s >> 30)) + i;
        }
    }

    ~Random() {
    }

    unsigned int nextInt(void) {
        const unsigned int t = seed_[0] ^ (seed_[0] << 11);
        seed_[0] = seed_[1];
        seed_[1] = seed_[2];
        seed_[2] = seed_[3];
        return seed_[3] = (seed_[3] ^ (seed_[3] >> 19)) ^ (t ^ (t >> 8));
    }

    double nextReal() {
        return (double)nextInt() * coefficient_;
    }

    double normal() {
        const double r1 = nextReal();
        const double r2 = nextReal();
        return sqrt(-2.0 * log(r1)) * sin(2.0 * PI * r2);
    }

};

#endif  // _RANDOM_H_
