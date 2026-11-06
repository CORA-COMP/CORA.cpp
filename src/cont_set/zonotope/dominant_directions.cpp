// dominant_directions - n generators whose parallelotope tightly encloses the zonotope
//
// Both filter lengths are cut to the number of nonzero generators. A singular P scores infinity
// and the first subset wins a tie; the columns come in the order of the length filter. Not for a
// batch; fewer than n nonzero generators raise degenerate_set.
//
// Syntax:   S = Z.dominant_directions();   S = Z.dominant_directions(length1, length2);
// Inputs:   filter_length1 - how many of the longest generators are considered (CORA: n + 5)
//           filter_length2 - how many n-subsets of largest |det| are scored (CORA: n + 3)
// Outputs:  S - (n, n): the columns of the subset P for which P box(P^-1 Z) has the least volume
// See also: reduce, box
//
// CORA: contSet/@zonotope/dominantDirections.m
// CORA: contSet/@zonotope/private/priv_lengthFilter.m
// CORA: contSet/@zonotope/private/priv_generatorVolumeFilter.m
// CORA: contSet/@zonotope/private/priv_volumeFilter.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

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
    const Tensor tolerance = singular_values.slice(-2, 0, 1) * (double(n) * machine_eps(P.dtype()));
    const Tensor singular = singular_values.slice(-2, n - 1, n).less_equal(tolerance);
    const Tensor radius = P.inv().matmul(G).abs_sum(); // the box of P^-1 Z, (B, n, 1)
    const Tensor volume = priv_abs_det(P).multiply(radius.prod(-2, true));
    return Tensor::where(singular,
                         volume.full_like(volume.shape(), std::numeric_limits<double>::infinity()),
                         volume);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::dominant_directions(std::optional<int> length1, std::optional<int> length2) const {
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
        return "dominant_directions: " + std::to_string(count) +
               " nonzero generators span "
               "fewer than the " +
               std::to_string(n) + " dimensions of the zonotope";
    });

    // the longest generators, longest first (priv_lengthFilter)
    const int64_t keep = std::min<int64_t>(filter_length1, count);
    const Tensor lengths = Gd.vector_norm(Tensor::Norm::two, -2).reshape({count});
    const Tensor longest = lengths.topk(keep).second;

    // the n-subsets of largest |det| (priv_generatorVolumeFilter), as indices into G
    const std::vector<int64_t> subsets =
        Combinations(keep, n).next(std::numeric_limits<int64_t>::max());
    const int64_t number = int64_t(subsets.size()) / n;
    const Tensor members = longest.take(subsets, 0).reshape({number, n});
    const Tensor candidates = priv_column_subsets(Gd, members);
    const int64_t shortlist = std::min<int64_t>({filter_length2, count, number});
    const Tensor volume = priv_abs_det(candidates).reshape({number});
    const Tensor best = volume.topk(shortlist).second;

    // the one of least volume (priv_volumeFilter): a tie goes to the first
    const Tensor scores = aux_volumes(candidates.take(best, 0), Gd).reshape({shortlist});
    const Tensor chosen = members.take(best, 0).take(scores.argmin(0, true), 0).reshape({n});
    return G.take(chosen, -1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
