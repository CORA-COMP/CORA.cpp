// gram_schmidt - an orthonormal basis that starts with a given direction, as CORA's gramSchmidt
//
// The columns are a and the unit axes but the one a points to the most; classical Gram-Schmidt
// orthonormalizes them in that order, the dot products summed from the first entry.
//
// Syntax:   std::vector<std::vector<double>> Q = gram_schmidt(a);
// Inputs:   a - a nonzero direction of n numbers
// Outputs:  Q - n orthonormal vectors, Q[0] = a / |a|
// See also: gram_schmidt.h
//
// CORA: global/functions/matlab/init/gramSchmidt.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "global/gram_schmidt.h"

#include <cmath>
#include <cstddef>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Row = std::vector<double>;

/// The dot product, summed from the first entry.
double aux_dot(const Row &a, const Row &b) {
    double sum = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) sum += a[k] * b[k];
    return sum;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::vector<Row> gram_schmidt(const Row &a) {
    const std::size_t n = a.size();
    // the columns a, then the axes but the one a points to the most
    std::size_t skip = 0;
    for (std::size_t i = 1; i < n; ++i)
        if (std::abs(a[i]) > std::abs(a[skip])) skip = i;
    std::vector<Row> B = {a};
    for (std::size_t i = 0; i < n; ++i) {
        if (i == skip) continue;
        Row e(n, 0.0);
        e[i] = 1.0;
        B.push_back(e);
    }
    // classical Gram-Schmidt on the columns, as CORA's gramSchmidt
    std::vector<Row> Q;
    for (std::size_t j = 0; j < n; ++j) {
        Row v = B[j];
        for (const Row &q : Q) {
            const double r = aux_dot(q, B[j]);
            for (std::size_t i = 0; i < n; ++i) v[i] -= r * q[i];
        }
        const double norm = std::sqrt(aux_dot(v, v));
        for (double &x : v) x /= norm;
        Q.push_back(v);
    }
    return Q;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
