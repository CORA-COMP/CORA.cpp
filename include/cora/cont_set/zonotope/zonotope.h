// zonotope - the class of zonotopes Z = {c + G b | |b|_inf <= 1}, as CORA's zonotope
//
// The center c is a column (..., n, 1); the generators G are (..., n, m), one per column.
// Immutable: every operation returns a new zonotope.
//
// Syntax:     Zonotope Z(c, G);   Zonotope Z = Zonotope::generate_random(n, m, rng);
// Operators:  M * Z (linear_map), s * Z, Z + Z2 and Z + I (plus), Z + v, Z - v, -Z, as in CORA
// Operations: linear_map, plus, linear_combination, support_function, interval, rand_point,
//             generate_random, stack, project, vertices, reduce, display (one file each, as in
//             CORA's @zonotope); the constructor checks its input (zonotope.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@zonotope/zonotope.m
//
// Not the competition's zonotope (competition/sets): that one is tuned for speed on one library.

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Zonotope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::zonotope;

    /// The zonotope of the center c (..., n, 1) and the generators G (..., n, m).
    Zonotope(Tensor c, Tensor G);

    /// A random zonotope with m generators in n dimensions (CORA: center 10*randn, generators of
    /// uniformly random direction and length ~ U[0, 1]).
    static Zonotope generate_random(int64_t n, int64_t m, Rng &rng);

    /// The center c and the generators G.
    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }

    using ContSet::rand_point;

    /// N random points c + G b, columns (..., n, N): b ~ U[-1, 1]^m for type "standard", b from
    /// the corners {-1, 1}^m (points on the boundary) for "extreme".
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

    /// M * Z for a matrix M (..., k, n): exact, c -> Mc and G -> MG.
    Zonotope linear_map(const Tensor &M) const;

    /// [M] * Z for an interval matrix (an Interval with (..., n, n) bounds): an enclosure.
    Zonotope linear_map(const Interval &M) const;

    /// The Minkowski sum: the centers add, the generators are joined.
    Zonotope plus(const Zonotope &Z2) const;

    /// The Minkowski sum with a box: its center adds, its radius becomes n axis generators.
    Zonotope plus(const Interval &I) const;

    /// A batch of zonotopes of one dimension (and batch shape) as one; those with fewer generators
    /// than the most are padded with zero generators.
    static Zonotope stack(const std::vector<Zonotope> &Zs);

    /// A zonotope of at most order * n generators (n the dimension) that contains this one, by
    /// Girard's method: the longest generators stay, the others become a box. Not a batch.
    Zonotope reduce(int order) const;

    /// Encloses the segments between matching points of Z and Z2 = M*Z + t (same generator
    /// factors): 2m + 1 generators.
    Zonotope linear_combination(const Zonotope &Z2) const;

  protected:
    int64_t dim_impl() const override { return c_.shape()[c_.shape().size() - 2]; }
    Tensor center_impl() const override { return c_; }
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Tensor rand_point_impl(int64_t N, Rng &rng, const std::string &type) const;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::string display_impl() const override;

  private:
    Tensor c_, G_;
};

// Operators, as CORA writes them: a matrix M * Z is the linear map, a number s * Z scales, Z + Z2
// is the Minkowski sum, and a column v translates.
inline Zonotope operator*(const Tensor &M, const Zonotope &Z) { return Z.linear_map(M); }
inline Zonotope operator*(const Interval &M, const Zonotope &Z) { return Z.linear_map(M); }
inline Zonotope operator*(double s, const Zonotope &Z) {
    return {Z.center() * s, Z.generators() * s};
}
inline Zonotope operator*(const Zonotope &Z, double s) { return s * Z; }
inline Zonotope operator+(const Zonotope &Z, const Zonotope &Z2) { return Z.plus(Z2); }
inline Zonotope operator+(const Zonotope &Z, const Interval &I) { return Z.plus(I); }
inline Zonotope operator+(const Interval &I, const Zonotope &Z) { return Z.plus(I); }
inline Zonotope operator+(const Zonotope &Z, const Tensor &v) { return {Z.center() + v, Z.generators()}; }
inline Zonotope operator+(const Tensor &v, const Zonotope &Z) { return Z + v; }
inline Zonotope operator-(const Zonotope &Z, const Tensor &v) { return {Z.center() - v, Z.generators()}; }
inline Zonotope operator-(const Zonotope &Z) { return {Z.center() * -1.0, Z.generators()}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
