// `I = {x | inf <= x <= sup}` elementwise, as CORA's `interval`, on CoraTensor.
//
// As in CORA the bounds may be matrices `(..., n, n)`: an interval matrix is an `Interval`
// (the correction matrix of linearSys is one), and `Zonotope::mtimes` takes it. `center` and
// `rad` are elementwise either way; `support_func`, `mtimes` and `contains` are for boxes,
// whose bounds are columns.

#pragma once

#include "contSet/contSet.h"

namespace cora::ct {

class Interval : public ContSet {
  public:
    /// The bounds: columns `(..., n, 1)` for a box, matrices `(..., n, n)` for an interval
    /// matrix.
    Tensor inf, sup;

    Interval(Tensor inf, Tensor sup) : inf(std::move(inf)), sup(std::move(sup)) {}

    /// CORA's `generateRandom`: centre `U[-2, 2]`, radius `R/2 · u` with `R ~ U[0, 10]` and
    /// `u ~ U[0, 1]`.
    static Interval generate_random(int64_t n, Rng &rng);

    int64_t dim() const override { return inf.shape()[inf.shape().size() - 2]; }
    Tensor center() const override { return (inf + sup) * 0.5; }
    Tensor support_func(const Tensor &d) const override;
    Interval interval() const override { return *this; }
    /// Uniform in the box.
    Tensor rand_point(int64_t N, Rng &rng) const override;

    /// The radius, of the shape of the bounds.
    Tensor rad() const { return (sup - inf) * 0.5; }

    /// `M * I` as CORA computes it, the interval hull: centres map by `M`, radii by `|M|`.
    Interval mtimes(const Tensor &M) const;

    /// The Minkowski sum.
    Interval plus(const Interval &I2) const { return {inf + I2.inf, sup + I2.sup}; }

    /// Whether the point `p` `(..., n, 1)` is in the box; for a batch, whether all are.
    bool contains(const Tensor &p) const;
};

} // namespace cora::ct
