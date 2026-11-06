// generate_random - a random linear system, as CORA's linearSys.generateRandom
//
// A = P J P^-1 for a random matrix P and a block-diagonal J whose eigenvalues are real numbers
// from real_interval and complex pairs with real part from it and imaginary part up to
// imaginary_max; B and C are Gaussian. The number of complex pairs is uniform in 0..n/2.
//
// Syntax:   LinearSys sys = LinearSys::generate_random(rng);
//           LinearSys::generate_random(rng, 5, 2, 1);
// Inputs:   rng - random generator
//           state_dimension, input_dimension, output_dimension - n, m, p, default: random
//           real_interval, imaginary_interval - boxes of one dimension, default: random
// Outputs:  sys - system x' = A x + B u, y = C x
// See also: linear_sys
//
// CORA: contDynamics/@linearSys/generateRandom.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Draws -------------------------------------------------------------------------------------------

/// A uniform number in [low, high].
double aux_uniform(Rng &rng, double low, double high) {
    double x;
    rng.uniform(&x, 1, low, high);
    return x;
}

/// A uniform integer in [low, high].
int64_t aux_integer(Rng &rng, int64_t low, int64_t high) {
    return std::min(
        high, low + static_cast<int64_t>(std::floor(aux_uniform(rng, 0, 1) * (high - low + 1))));
}

/// A matrix of Gaussian entries.
CORA_FROM_NUMBERS Tensor aux_normal(Rng &rng, int64_t rows, int64_t cols) {
    std::vector<double> data(rows * cols);
    rng.normal(data.data(), data.size(), 1.0);
    return Tensor::from_data(data, {rows, cols});
}

/// The Jordan form: real eigenvalues on the diagonal, a rotation block per complex pair.
CORA_FROM_NUMBERS Tensor aux_jordan(Rng &rng, int64_t n, double real_low, double real_high,
                                    double imaginary_max) {
    const int64_t pairs = aux_integer(rng, 0, n / 2);
    std::vector<Tensor> blocks;
    if (n - 2 * pairs > 0) {
        std::vector<double> diagonal(n - 2 * pairs);
        for (double &x : diagonal) x = aux_uniform(rng, real_low, real_high);
        blocks.push_back(Tensor::from_data(diagonal, {n - 2 * pairs, 1}).diag());
    }
    std::vector<double> real(pairs), imaginary(pairs);
    for (double &x : real) x = aux_uniform(rng, real_low, real_high);
    for (double &x : imaginary) x = aux_uniform(rng, 0, imaginary_max);
    for (int64_t i = 0; i < pairs; ++i)
        blocks.push_back(
            Tensor::from_data({real[i], -imaginary[i], imaginary[i], real[i]}, {2, 2}));
    return Tensor::block_diag(blocks);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The system --------------------------------------------------------------------------------------

LinearSys LinearSys::generate_random(Rng &rng, std::optional<int64_t> state_dimension,
                                     std::optional<int64_t> input_dimension,
                                     std::optional<int64_t> output_dimension,
                                     const std::optional<Interval> &real_interval,
                                     const std::optional<Interval> &imaginary_interval) {
    detail::Entry entry;
    CORA_CHECK((!state_dimension || *state_dimension >= 1) &&
                   (!input_dimension || *input_dimension >= 1) &&
                   (!output_dimension || *output_dimension >= 1),
               Error::wrong_value,
               [] { return std::string("generate_random: the dimensions must be at least 1"); });
    CORA_CHECK((!real_interval || real_interval->dim() == 1) &&
                   (!imaginary_interval || imaginary_interval->dim() == 1),
               Error::wrong_value, [] {
                   return std::string("generate_random: real_interval and imaginary_interval "
                                      "must be boxes of one dimension");
               });
    const int64_t n = state_dimension ? *state_dimension : aux_integer(rng, 4, 10);
    const int64_t m = input_dimension ? *input_dimension : aux_integer(rng, 1, 3);
    const int64_t p = output_dimension ? *output_dimension : aux_integer(rng, 1, 2);
    // the defaults draw the real bound first, as CORA does
    double real_low, real_high, imaginary_max;
    if (real_interval) {
        real_low = real_interval->inf().item();
        real_high = real_interval->sup().item();
    } else {
        real_low = -1 - 10 * aux_uniform(rng, 0, 1);
        real_high = -aux_uniform(rng, 0, 1);
    }
    if (imaginary_interval) {
        imaginary_max =
            std::min(std::abs(imaginary_interval->inf().item()), imaginary_interval->sup().item());
    } else {
        imaginary_max = 10 * aux_uniform(rng, 0, 1);
    }
    // A = P J P^-1 has the eigenvalues of J, B and C are Gaussian
    const Tensor J = aux_jordan(rng, n, real_low, real_high, imaginary_max);
    const Tensor P = aux_normal(rng, n, n);
    const Tensor A = P.matmul(J.matmul(P.inv()));
    return {A, aux_normal(rng, n, m), aux_normal(rng, p, n)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
