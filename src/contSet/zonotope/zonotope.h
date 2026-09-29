// `Z = {c + G b | ||b||_inf <= 1}`, as CORA's `zonotope`, on CoraTensor. Names follow CORA:
// `c` and `G` for centre and generators, `mtimes`, `plus`, `linComb`, `supportFunc`.
//
// Not to be confused with the competition's own zonotope in competition/sets: those are
// the catalog's operations, laid out for speed on one library each.

#pragma once

#include "contSet/contSet.h"
#include "contSet/interval/interval.h"

namespace cora {
class Rng;
}

namespace cora::ct {

class Zonotope : public ContSet {
  public:
    /// The centre `c`, a column `(..., n, 1)`, and the generators `G` `(..., n, m)`, one
    /// per column.
    Tensor c, G;

    Zonotope(Tensor c, Tensor G) : c(std::move(c)), G(std::move(G)) {}

    /// CORA's `generateRandom`: centre `10 · randn`, `m` generators, each a uniformly
    /// random unit direction with a length `~ U[0, 1]`.
    static Zonotope generate_random(int64_t n, int64_t m, Rng &rng);

    int64_t dim() const override { return c.shape()[c.shape().size() - 2]; }
    Tensor center() const override { return c; }
    Tensor support_func(const Tensor &d) const override;
    Interval interval() const override;

    /// `M * Z`.
    Zonotope mtimes(const Tensor &M) const;

    /// `[M] * Z` for an interval matrix `[M]`, an `Interval` whose bounds are `(..., n, n)`
    /// matrices as in CORA; enclosed by a zonotope: the centre matrix maps the set, and one
    /// axis-aligned generator per dimension covers what the radius adds.
    Zonotope mtimes(const Interval &M) const;

    /// The Minkowski sum.
    Zonotope plus(const Zonotope &Z2) const;

    /// CORA's `linComb` for a set that shares its generator factors with this one (`Z2` is
    /// `M*Z + t`): encloses the segments between matching points, with `2m + 1` generators.
    Zonotope lin_comb(const Zonotope &Z2) const;
};

} // namespace cora::ct
