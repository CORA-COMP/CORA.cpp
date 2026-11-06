// generate_random - a random ellipsoid, as CORA's ellipsoid.generateRandom
//
// Q = A'A for a Gaussian A (n, n), its singular values below sqrt(tol) of the largest lifted, one
// to three of them set to 0 for a degenerate ellipsoid, and Q scaled to entries below 10. The
// dimension is given, else the length of the center, else uniform in 1..30; the center is N(0,
// 1)^n.
//
// Syntax:   E = Ellipsoid::generate_random(rng);   E = Ellipsoid::generate_random(rng, n,
// is_degenerate, center); Inputs:   rng - the generator of the random numbers
//           n, is_degenerate, center - optional, as CORA's name-value pairs Dimension,
//           IsDegenerate, Center
// Outputs:  E - the ellipsoid
// See also: Zonotope::generate_random
//
// CORA: contSet/@ellipsoid/generateRandom.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Ellipsoid Ellipsoid::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                       bool is_degenerate,
                                                       const std::optional<Tensor> &center) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!center || (center->ndim() == 2 && center->shape().back() == 1), Error::wrong_value,
               [] { return std::string("generate_random: the center must be a column (n, 1)"); });
    CORA_CHECK(!(dimension && center) || center->shape()[0] == *dimension,
               Error::dimension_mismatch,
               [] { return std::string("generate_random: the center and the dimension differ"); });
    constexpr int64_t kMaxDimension = 30; // CORA's maxdim
    const int64_t n = dimension ? *dimension
                      : center  ? center->shape()[0]
                                : priv_random_dimension(rng, kMaxDimension);
    const auto draw = [&](int64_t rows, int64_t cols) {
        std::vector<double> values(static_cast<std::size_t>(rows * cols));
        rng.normal(values.data(), values.size(), 1.0);
        return Tensor::from_data(values, {rows, cols});
    };
    const Tensor q = center ? *center : draw(n, 1);
    const Tensor A = draw(n, n);
    const Tensor S = A.matrix_transpose().matmul(A);
    // singular values below sqrt(tol) of the largest are lifted, so Q is not degenerate by accident
    const std::array<Tensor, 3> usv = ((S + S.matrix_transpose()) * 0.5).svd();
    std::vector<double> s = usv[1].data();
    const double largest = s.front(),
                 smallest =
                     std::sqrt(resolve_tol(entry, std::nullopt, S.dtype(), "generate_random")) *
                     largest;
    std::vector<double> u(s.size());
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    for (std::size_t i = 0; i < s.size(); ++i)
        if (s[i] < smallest) s[i] = largest + 0.5 * (largest - smallest) * u[i];
    if (is_degenerate) {
        // randi(n) of the values are zeroed, at randi(n_d) positions drawn with repetition
        double draw_u = 0.0;
        rng.uniform(&draw_u, 1, 0.0, 1.0);
        const int64_t zeroed = std::min<int64_t>(
            n, 1 + static_cast<int64_t>(std::floor(static_cast<double>(n) * draw_u)));
        for (int64_t k = 0; k < zeroed; ++k) {
            rng.uniform(&draw_u, 1, 0.0, 1.0);
            s[static_cast<std::size_t>(std::min<int64_t>(
                zeroed - 1,
                static_cast<int64_t>(std::floor(static_cast<double>(zeroed) * draw_u))))] = 0.0;
        }
    }
    Tensor Q = usv[0].multiply(Tensor::from_data(s, {1, n})).matmul(usv[0].matrix_transpose());
    Q = (Q + Q.matrix_transpose()) * 0.5;
    // entries beyond 10 in magnitude: scale the largest to 10
    const double biggest = std::max(std::abs(Q.min().item()), Q.max().item());
    constexpr double kMaxAbsValue = 10.0;
    if (Q.abs().max().item() > kMaxAbsValue) Q = Q * (kMaxAbsValue / biggest);
    return {Q, q};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
