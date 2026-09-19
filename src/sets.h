// The set representations, as Eigen matrices.
//
// Every set holds a batch of `B` sets (`1` when the instance is not batched), laid out so
// that one call covers the whole batch: an `Interval` keeps `lo`/`hi` as `n × B`, one set
// per column, and a `Zonotope` a centre `n × B` and the batch's generator blocks side by
// side in one `n × (m·B)` matrix, one generator per column. Point clouds are
// `n × (points·B)`, one point per column.
//
// Everything is a template on the scalar type. `double` is what the catalog measures and
// what Eigen vectorizes; the same source instantiates for a differentiable scalar
// (`Eigen::AutoDiffScalar`, autodiff's `dual`, CoDiPack, …) so the library can be
// differentiated through, at the usual cost of leaving the vectorized paths behind.

#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "rng.h"

namespace cora {

template <typename T>
using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

/// The plain value of a scalar, for the parts that are not differentiable — the
/// containment LP and the verdicts. A differentiable scalar overloads this.
inline double value_of(double x) { return x; }

/// An axis-aligned box, `{x | lo <= x <= hi}` elementwise.
template <typename T>
struct Interval {
    Mat<T> lo, hi;

    Eigen::Index dim() const { return lo.rows(); }
    Eigen::Index batch() const { return lo.cols(); }

    Mat<T> center() const { return (hi + lo) * T(0.5); }
    Mat<T> radius() const { return (hi - lo) * T(0.5); }
};

/// `{c + G b | ||b||_inf <= 1}`, one generator per column of `G`.
template <typename T>
struct Zonotope {
    Mat<T> c;
    Mat<T> g;
    Eigen::Index m = 0;

    Eigen::Index dim() const { return c.rows(); }
    Eigen::Index batch() const { return c.cols(); }

    /// The generators of set `b`.
    auto block(Eigen::Index b) const { return g.middleCols(b * m, m); }
    auto block(Eigen::Index b) { return g.middleCols(b * m, m); }
};

/// The catalog's set names.
inline constexpr const char *kRepresentations[] = {"interval", "zonotope"};

// ---- construction ----------------------------------------------------------------

/// CORA's `interval.generateRandom`: centre `U[-2, 2]`, radius `R/2 · u` with
/// `R ~ U[0, 10]` and `u ~ U[0, 1]`.
Interval<double> random_interval(Rng &rng, Eigen::Index n, Eigen::Index batch);

/// CORA's `zonotope.generateRandom`: centre `10 · randn`, `m` generators, each a
/// uniformly random unit direction with a length `~ U[0, 1]`.
Zonotope<double> random_zonotope(Rng &rng, Eigen::Index n, Eigen::Index m,
                                 Eigen::Index batch);

/// The unit zonotope at the origin: the `startup` instance.
template <typename T>
Zonotope<T> origin(Eigen::Index n) {
    return Zonotope<T>{Mat<T>::Zero(n, 1), Mat<T>::Identity(n, n), n};
}

// ---- operations ------------------------------------------------------------------

/// `points` points per set, uniform in the box; `n × (points·B)`.
Mat<double> rand_point(const Interval<double> &s, Eigen::Index points, Rng &rng);

/// `points` points per set, CORA's `standard`: `c + G b` with `b ~ U[-1, 1]^m`.
Mat<double> rand_point(const Zonotope<double> &s, Eigen::Index points, Rng &rng);

/// `max_{x in S} d'x` for one direction per set, `d` of size `n × B`; one value per set.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> support_func(const Interval<T> &s, const Mat<T> &d) {
    return ((s.center().array() * d.array()) + (s.radius().array() * d.array().abs()))
        .colwise()
        .sum()
        .transpose();
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> support_func(const Zonotope<T> &s, const Mat<T> &d) {
    Eigen::Matrix<T, Eigen::Dynamic, 1> out(s.batch());
#pragma omp parallel for schedule(static) if (s.batch() > 1)
    for (Eigen::Index b = 0; b < s.batch(); ++b) {
        out(b) = s.c.col(b).dot(d.col(b))
                 + (d.col(b).transpose() * s.block(b)).array().abs().sum();
    }
    return out;
}

/// `M · S` as CORA computes it, the interval hull: centres map by `M`, radii by `|M|`.
template <typename T>
Interval<T> mat_mul(const Interval<T> &s, const Mat<T> &m) {
    const Mat<T> c = m * s.center();
    const Mat<T> r = m.array().abs().matrix() * s.radius();
    return Interval<T>{c - r, c + r};
}

/// `M · S`; `M` maps every generator of every set alike, so the batch is one product.
template <typename T>
Zonotope<T> mat_mul(const Zonotope<T> &s, const Mat<T> &m) {
    return Zonotope<T>{m * s.c, m * s.g, s.m};
}

template <typename T>
Interval<T> mink_sum(const Interval<T> &a, const Interval<T> &b) {
    return Interval<T>{a.lo + b.lo, a.hi + b.hi};
}

template <typename T>
Zonotope<T> mink_sum(const Zonotope<T> &a, const Zonotope<T> &b) {
    Zonotope<T> out{a.c + b.c, Mat<T>(a.dim(), (a.m + b.m) * a.batch()), a.m + b.m};
#pragma omp parallel for schedule(static) if (a.batch() > 1)
    for (Eigen::Index k = 0; k < a.batch(); ++k) {
        out.g.middleCols(k * out.m, a.m) = a.block(k);
        out.g.middleCols(k * out.m + a.m, b.m) = b.block(k);
    }
    return out;
}

/// One answer per point, `1` for inside. A byte per point rather than `vector<bool>`,
/// whose packed bits cannot be written from several threads at once.
using Mask = std::vector<std::uint8_t>;

/// Whether each of the `points` points of each set lies in its box.
Mask contains(const Interval<double> &s, const Mat<double> &p, Eigen::Index points);

/// Whether each of the `points` points of each set lies in the zonotope — exactly.
Mask contains(const Zonotope<double> &s, const Mat<double> &p, Eigen::Index points);

} // namespace cora
