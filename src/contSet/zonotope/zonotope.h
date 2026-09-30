// zonotope - the class of zonotopes Z = {c + G b | |b|_inf <= 1}, as CORA's zonotope
//
// The center c is a column (..., n, 1); the generators G are (..., n, m), one per column.
//
// Syntax:     Zonotope Z(c, G);   Zonotope Z = Zonotope::generateRandom(n, m, rng);
// Operators:  M * Z (mtimes), s * Z, Z + Z2 (plus), Z + v, Z - v, -Z, as in CORA
// Operations: mtimes, plus, linComb, supportFunc, interval, randPoint, generateRandom
//             stack, project, vertices, reduce, display (one file each, as in CORA's @zonotope)
// See also:   contSet/contSet.h, contSet/interval/interval.h
//
// Not the competition's zonotope (competition/sets): that one is tuned for speed on one library.

#pragma once

#include "contSet/contSet.h"
#include "contSet/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Zonotope : public ContSet {
  public:
    /// The center (..., n, 1) and the generators (..., n, m).
    Tensor c, G;

    Zonotope(Tensor c, Tensor G) : c(std::move(c)), G(std::move(G)) {}

    /// A random zonotope with m generators in n dimensions (CORA: center 10*randn, generators of
    /// uniformly random direction and length ~ U[0, 1]).
    static Zonotope generateRandom(int64_t n, int64_t m, Rng &rng);

    /// The dimension: the rows of the center.
    int64_t dim() const override { return c.shape()[c.shape().size() - 2]; }

    /// The center c.
    Tensor center() const override { return c; }

    /// max_{x in Z} d'x = d'c + |G'd|_1; d is a column (..., n, 1).
    Tensor supportFunc(const Tensor &d) const override;

    /// The smallest box containing the zonotope: c -/+ sum_j |G_j|.
    Interval interval() const override;

    /// N random points c + G b with b ~ U[-1, 1]^m ("standard"), columns (..., n, N).
    Tensor randPoint(int64_t N, Rng &rng) const override { return randPoint(N, rng, "standard"); }

    /// As above; type "extreme" takes b from the corners {-1, 1}^m, points on the boundary.
    Tensor randPoint(int64_t N, Rng &rng, const std::string &type) const;

    /// M * Z for a matrix M (..., n, n): exact, c -> Mc and G -> MG.
    Zonotope mtimes(const Tensor &M) const;

    /// [M] * Z for an interval matrix (an Interval with (..., n, n) bounds): an enclosure.
    Zonotope mtimes(const Interval &M) const;

    /// The Minkowski sum: the centers add, the generators are joined.
    Zonotope plus(const Zonotope &Z2) const;

    /// A batch of zonotopes of one dimension (and batch shape) as one; those with fewer generators
    /// than the most are padded with zero generators.
    static Zonotope stack(const std::vector<Zonotope> &Zs);

    /// The projection onto `dims`: the selected rows of c and G, exact.
    std::unique_ptr<ContSet> project(const std::vector<int64_t> &dims) const override;

    /// The corners of a two-dimensional zonotope, counter-clockwise, per batch member.
    std::vector<Polygon> vertices() const override;

    /// The dimension, center and generators as text.
    std::string display() const override;

    /// A zonotope of at most order * n generators (n the dimension) that contains this one: the
    /// longest generators stay, the others become a box. A single zonotope, not a batch.
    Zonotope reduce(int order) const;

    /// Encloses the segments between matching points of Z and Z2 = M*Z + t (same generator
    /// factors): 2m + 1 generators.
    Zonotope linComb(const Zonotope &Z2) const;
};

// Operators, as CORA writes them: a matrix M * Z is the linear map, a number s * Z scales, Z + Z2
// is the Minkowski sum, and a column v translates.
inline Zonotope operator*(const Tensor &M, const Zonotope &Z) { return Z.mtimes(M); }
inline Zonotope operator*(const Interval &M, const Zonotope &Z) { return Z.mtimes(M); }
inline Zonotope operator*(double s, const Zonotope &Z) { return {Z.c * s, Z.G * s}; }
inline Zonotope operator*(const Zonotope &Z, double s) { return s * Z; }
inline Zonotope operator+(const Zonotope &Z, const Zonotope &Z2) { return Z.plus(Z2); }
inline Zonotope operator+(const Zonotope &Z, const Tensor &v) { return {Z.c + v, Z.G}; }
inline Zonotope operator+(const Tensor &v, const Zonotope &Z) { return Z + v; }
inline Zonotope operator-(const Zonotope &Z, const Tensor &v) { return {Z.c - v, Z.G}; }
inline Zonotope operator-(const Zonotope &Z) { return {Z.c * -1.0, Z.G}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
