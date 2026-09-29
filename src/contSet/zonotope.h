// The zonotope and interval matrix written against CoraTensor, so they run on every
// backend and batch wherever the backend does. Names follow CORA: `c` and `G` for centre
// and generators, `mtimes`, `plus`, `linComb`, `supportFunc`.
//
// Not to be confused with sets.h and torch_sets.h: those are the catalog's per-backend
// implementations, laid out for speed on one library each.

#pragma once

#include "tensor/tensor.h"

namespace cora::ct {

/// A matrix of intervals, `{M | lo <= M <= hi}` elementwise, `(..., n, n)`, kept as centre
/// and radius.
struct IntervalMatrix {
    Tensor center, radius;

    static IntervalMatrix from_bounds(const Tensor &lo, const Tensor &hi) {
        return {(lo + hi) * 0.5, (hi - lo) * 0.5};
    }
};

/// `Z = {c + G b | ||b||_inf <= 1}` with the centre `c` a column `(..., n, 1)` and the
/// generators `G` `(..., n, m)`, one per column.
struct Zonotope {
    Tensor c, G;

    /// `M * Z`.
    Zonotope mtimes(const Tensor &M) const;

    /// `[M] * Z`, enclosed by a zonotope: the centre matrix maps the set, and one
    /// axis-aligned generator per dimension covers what the radius adds.
    Zonotope mtimes(const IntervalMatrix &M) const;

    /// The Minkowski sum.
    Zonotope plus(const Zonotope &Z2) const;

    /// CORA's `linComb` for a set that shares its generator factors with this one (`Z2` is
    /// `M*Z + t`): encloses the segments between matching points, with `2m + 1` generators.
    Zonotope lin_comb(const Zonotope &Z2) const;

    /// `max_{x in Z} dᵀx` for a column `d` `(..., n, 1)`: a `(..., 1, 1)` tensor.
    Tensor support_func(const Tensor &d) const;
};

} // namespace cora::ct
