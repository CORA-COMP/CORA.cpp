// lift - a zonotope in a space of higher dimension, with its states at given positions
//
// Syntax:   Zl = Z.lift(N, dims);
// Inputs:   N - the dimension of the new space;  dims - the position (0-based) of each state of Z
//           in it; N positions, each used once: new dimensions would be unbounded, which no
//           zonotope represents
// Outputs:  Zl - the zonotope whose state dims[i] is state i of Z, exact (a permutation).
//           CORA's lift_ projects onto dims instead, which is the inverse permutation and the
//           same only where dims is its own inverse; CORA.cpp follows the meaning of lift
// See also: project
//
// CORA: contSet/@zonotope/lift_.m
// CORA: contSet/@contSet/lift.m

#include "cora/cont_set/zonotope/zonotope.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(dim() <= N, Error::wrong_value, [&] {
        return "lift: the dimension N = " + std::to_string(N) +
               " of the new space must be at least the dimension " + std::to_string(dim());
    });
    CORA_CHECK(int64_t(dims.size()) == dim(), Error::wrong_value, [&] {
        return "lift: dims needs one position per state, " + std::to_string(dim()) + ", got " +
               std::to_string(dims.size());
    });
    CORA_CHECK(dims.empty() || (*std::min_element(dims.begin(), dims.end()) >= 0 &&
                                *std::max_element(dims.begin(), dims.end()) < N),
               Error::wrong_value, [&] {
                   return "lift: dims must be positions 0..N-1 of the new space, N = " +
                          std::to_string(N);
               });
    CORA_CHECK(int64_t(dims.size()) == N, Error::not_defined, [] {
        return std::string("lift: the new dimensions cannot be unbounded, as a zonotope is "
                           "always bounded; dims must be a permutation of all N states");
    });
    // the state at position j of the result is the state i with dims[i] = j
    std::vector<int64_t> source(N, -1);
    for (int64_t i = 0; i < N; ++i) {
        CORA_CHECK(source[dims[i]] < 0, Error::wrong_value, [&] {
            return "lift: the position " + std::to_string(dims[i]) + " is used twice";
        });
        source[dims[i]] = i;
    }
    return {c_.take(source, -2), G_.take(source, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
