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


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, dims, "lift");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the rest is the class's own: it holds for lift_impl, whose caller has checked the above
    if (int64_t(dims.size()) != N)
        CORA_FAIL(Error::not_defined,
                  "lift: the new dimensions cannot be unbounded, as a zonotope is always "
                  "bounded; dims must be a permutation of all N states");
    // the state at position j of the result is the state i with dims[i] = j
    std::vector<int64_t> source(N, -1);
    for (int64_t i = 0; i < N; ++i) {
        if (source[dims[i]] >= 0)
            CORA_FAIL(Error::wrong_value,
                      "lift: the position " + std::to_string(dims[i]) + " is used twice");
        source[dims[i]] = i;
    }
    return {c_.take(source, -2), G_.take(source, -2)};
}

Set Zonotope::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
