// reduce - an enclosure of a polytope by few random halfspaces, as CORA's polytope.reduce
//
// The set is compacted; the empty set stays empty. The method "rand" draws 2 * order * n Gaussian
// directions and takes the support function of the set along each as the offset, so the result
// contains the set (a direction without bound drops its row). Not a batch.
//
// Syntax:   Pres = P.reduce(order, "rand");   Pres = P.reduce(order, "rand", &rng);
// Inputs:   order - a positive integer: the number of rows is 2 * order * n
//           method - "rand"
//           rng - draws the directions (default: a generator of the seed 0)
// Outputs:  Pres - a polytope with at most 2 * order * n rows that contains P
// See also: Polytope::compact, Polytope::support_function
//
// CORA: contSet/@polytope/reduce.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope Polytope::reduce(int order, const std::string &method, Rng *rng_arg) const {
    detail::Entry entry;
    CORA_CHECK(method == "rand", Error::wrong_value,
               [&] { return "reduce: the method must be rand, got '" + method + "'"; });
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the order must be a positive integer, got " + std::to_string(order);
    });
    // without a generator, a fixed seed gives the same directions every time
    Rng fallback(0);
    Rng &rng = rng_arg ? *rng_arg : fallback;
    const Polytope P = compact();
    if (P.representsa("emptySet")) return empty(dim());
    const Tensor &like = P.is_h_rep() ? P.A() : P.V();
    const int64_t n = dim(), rows = 2 * int64_t(order) * n;
    // the directions, then the support function of the set along each
    std::vector<double> directions(std::size_t(rows * n));
    rng.normal(directions.data(), directions.size(), 1.0);
    std::vector<double> offsets;
    for (int64_t i = 0; i < rows; ++i) {
        const Row direction(directions.begin() + i * n, directions.begin() + (i + 1) * n);
        offsets.push_back(P.support_function(priv_vector_tensor(like, direction)).item());
    }
    return {Tensor::like(like, directions, {rows, n}), Tensor::like(like, offsets, {rows})};
}

Set Polytope::reduce_impl(int order) const { return reduce(order, "rand"); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
