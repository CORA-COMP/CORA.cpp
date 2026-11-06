// priv_gram_schmidt - an orthonormal basis that starts with a given direction
//
// CORA's gramSchmidt of a vector: the vector followed by the unit vectors without the one of its
// largest entry, orthonormalized one after the other (classical Gram-Schmidt).
//
// Syntax:   std::vector<double> Q = priv_gram_schmidt(v);
// Inputs:   v - a nonzero vector (p entries)
// Outputs:  Q - the basis as columns, row-major (p, p); the first column is v / |v|
// See also: ConZonotope::zono_bundle, Zonotope::simplex
//
// CORA.cpp only: CORA's gramSchmidt is a helper of its global functions
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_gram_schmidt.h"
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<double> priv_gram_schmidt(const std::vector<double> &v) {
    // the unit vector of the largest entry of v is the one left out of the basis
    const std::size_t p = v.size();
    std::size_t largest = 0;
    for (std::size_t i = 1; i < p; ++i)
        if (std::abs(v[i]) > std::abs(v[largest])) largest = i;
    // the input columns B: v, then the unit vectors but the largest one
    std::vector<std::vector<double>> B{v}, Q;
    for (std::size_t k = 0; k < p; ++k) {
        if (k == largest) continue;
        B.emplace_back(p, 0.0);
        B.back()[k] = 1.0;
    }
    // classical Gram-Schmidt: remove the parts along the earlier columns, then normalize
    for (std::size_t j = 0; j < p; ++j) {
        std::vector<double> w = B[j];
        for (std::size_t i = 0; i < j; ++i) {
            double r = 0.0;
            for (std::size_t k = 0; k < p; ++k) r += Q[i][k] * B[j][k];
            for (std::size_t k = 0; k < p; ++k) w[k] -= r * Q[i][k];
        }
        double norm = 0.0;
        for (const double x : w) norm += x * x;
        norm = std::sqrt(norm);
        for (double &x : w) x /= norm;
        Q.push_back(w);
    }
    std::vector<double> out(p * p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) out[i * p + j] = Q[j][i];
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
