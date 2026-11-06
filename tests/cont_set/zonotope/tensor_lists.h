// tensor_lists - what the tests of tensor_multiplication share: CORA's lists and random tensors
//
// Syntax:   Zonotope::TensorLists list = test::triples(3);   Tensor M = test::random_tensor(rng,
// n); Outputs:  list - the ordered triples of m generators (perm), the sorted ones (comb) and the
//           column of the sorted triple of each (map);  M - a normal random tensor (n, n, n, n)
// See also: test_zonotope_tensor_multiplication.cpp
//
// CORA.cpp only: CORA builds options.list outside the function
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"

#include <algorithm>
#include <array>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace test {

/// CORA's lists for m generators.
inline cora::Zonotope::TensorLists triples(int m) {
    cora::Zonotope::TensorLists list;
    std::vector<std::array<int, 3>> sorted;
    for (int a = 0; a < m; ++a)
        for (int b = a; b < m; ++b)
            for (int c = b; c < m; ++c) {
                sorted.push_back({a, b, c});
                list.comb.push_back({a, b, c});
            }
    for (int a = 0; a < m; ++a)
        for (int b = 0; b < m; ++b)
            for (int c = 0; c < m; ++c) {
                std::array<int, 3> key{a, b, c};
                std::sort(key.begin(), key.end());
                list.perm.push_back({a, b, c});
                list.map.push_back(std::find(sorted.begin(), sorted.end(), key) - sorted.begin());
            }
    return list;
}

/// A random tensor (n, n, n, n) of standard normal entries.
inline cora::Tensor random_tensor(cora::Rng &rng, int n) {
    std::vector<double> data(n * n * n * n);
    rng.normal(data.data(), data.size(), 1.0);
    return cora::Tensor::from_data(data, {n, n, n, n});
}

} // namespace test

// ---------------------------------------  END OF CODE  ---------------------------------------- //
