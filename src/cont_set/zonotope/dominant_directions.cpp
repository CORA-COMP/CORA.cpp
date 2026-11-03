// dominant_directions - n generators whose parallelotope tightly encloses the zonotope
//
// Syntax:   S = Z.dominant_directions();   S = Z.dominant_directions(length1, length2);
// Inputs:   filter_length1 - how many of the longest generators are considered (CORA: n + 5)
//           filter_length2 - how many n-subsets of largest |det| are scored (CORA: n + 3)
//           both are cut to the number of nonzero generators
// Outputs:  S - (n, n): the columns of the subset P for which P box(P^-1 Z) has the least volume
//           (a singular P scores infinity; the first subset wins a tie), in the order of the
//           length filter (longest first). Not for a batch; fewer than n nonzero generators
//           raise degenerate_set
// See also: reduce, box
//
// CORA: contSet/@zonotope/dominantDirections.m
// CORA: contSet/@zonotope/private/priv_lengthFilter.m
// CORA: contSet/@zonotope/private/priv_generatorVolumeFilter.m
// CORA: contSet/@zonotope/private/priv_volumeFilter.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"
#include "cont_set/zonotope/private/priv.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Subsets -----------------------------------------------------------------------------------------

/// The n-subsets of {0, ..., total - 1} in lexicographic order, as a flat list (CORA's combinator).
std::vector<int64_t> aux_combinations(int64_t total, int64_t n) {
    std::vector<int64_t> flat, current(n);
    for (int64_t i = 0; i < n; ++i) current[i] = i;
    while (true) {
        flat.insert(flat.end(), current.begin(), current.end());
        int64_t i = n - 1;
        while (i >= 0 && current[i] == total - n + i) --i;
        if (i < 0) return flat;
        ++current[i];
        for (int64_t j = i + 1; j < n; ++j) current[j] = current[j - 1] + 1;
    }
}

/// The n x n matrices (B, n, n) whose columns are G(:, subset) for every row of `members` (B, n).
Tensor aux_subsets(const Tensor &G, const Tensor &members) {
    const int64_t B = members.shape()[0], n = members.shape()[1];
    return G.take(members.reshape({B * n}), -1).reshape({n, B, n}).permute_dims({1, 0, 2});
}

/// |det| of matrices (B, n, n), by the diagonal of R of a QR factorization: (B, 1, 1).
Tensor aux_abs_det(const Tensor &P) { return P.qr().second.diagonal().prod(-2, true); }

/// The generators of the zonotope that are not zero (CORA's nonzeroFilter without a tolerance).
Tensor aux_nonzero_generators(const Tensor &G) {
    const Tensor values = G.detach();
    const Tensor present =
        values.not_equal(values.zeros_like()).any(-2).reshape({G.shape().back()});
    return priv_select_generators(G, present.nonzero());
}

// Volumes -----------------------------------------------------------------------------------------

/// The volume of P box(P^-1 Z) up to the factor 2^n, for the matrices P (B, n, n): infinity if P is
/// singular (CORA's priv_volumeFilter).
Tensor aux_volumes(const Tensor &P, const Tensor &G) {
    const int64_t n = P.shape().back();
    const Tensor singular_values = P.svd()[1]; // (B, n, 1), descending
    const Tensor tolerance = singular_values.slice(-2, 0, 1) *
                             (double(n) * std::numeric_limits<double>::epsilon());
    const Tensor singular = singular_values.slice(-2, n - 1, n).less_equal(tolerance);
    const Tensor radius = P.inv().matmul(G).abs_sum();      // the box of P^-1 Z, (B, n, 1)
    const Tensor volume = aux_abs_det(P).multiply(radius.prod(-2, true));
    return Tensor::where(singular, volume.full_like(volume.shape(),
                                                    std::numeric_limits<double>::infinity()),
                         volume);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::dominant_directions(std::optional<int> length1,
                                     std::optional<int> length2) const {
    detail::Entry entry;
    const int64_t n = dim();
    const int filter_length1 = length1.value_or(int(n) + 5);
    const int filter_length2 = length2.value_or(int(n) + 3);
    CORA_CHECK(filter_length1 >= 1 && filter_length2 >= 1, Error::wrong_value, [&] {
        return "dominant_directions: the filter lengths must be at least 1, got " +
               std::to_string(filter_length1) + " and " + std::to_string(filter_length2);
    });
    CORA_CHECK(G_.ndim() == 2, Error::not_supported, [] {
        return std::string("dominant_directions: the choice of generators is made per zonotope; "
                           "it supports a single zonotope, not a batch");
    });
    // the choice is made on detached values; the result takes its columns from G, with gradients
    const Tensor G = aux_nonzero_generators(G_);
    const Tensor Gd = G.detach();
    const int64_t count = G.shape().back();
    CORA_CHECK(count >= n, Error::degenerate_set, [&] {
        return "dominant_directions: " + std::to_string(count) + " nonzero generators span "
               "fewer than the " + std::to_string(n) + " dimensions of the zonotope";
    });

    // the longest generators, longest first (priv_lengthFilter)
    const int64_t keep = std::min<int64_t>(filter_length1, count);
    const Tensor lengths = Gd.vector_norm(Tensor::Norm::two, -2).reshape({count});
    const Tensor longest = lengths.topk(keep).second;

    // the n-subsets of largest |det| (priv_generatorVolumeFilter), as indices into G
    const std::vector<int64_t> subsets = aux_combinations(keep, n);
    const int64_t number = int64_t(subsets.size()) / n;
    const Tensor members = longest.take(subsets, 0).reshape({number, n});
    const Tensor candidates = aux_subsets(Gd, members);
    const int64_t shortlist = std::min<int64_t>({filter_length2, count, number});
    const Tensor volume = aux_abs_det(candidates).reshape({number});
    const Tensor best = volume.topk(shortlist).second;

    // the one of least volume (priv_volumeFilter): a tie goes to the first
    const Tensor scores = aux_volumes(candidates.take(best, 0), Gd).reshape({shortlist});
    const Tensor chosen = members.take(best, 0).take(scores.argmin(0, true), 0).reshape({n});
    return G.take(chosen, -1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
