// priv - the private functions of zonotope, as CORA's @zonotope/private
//
// Not for callers of the class: the operations that mix zonotopes with other sets use them.
//
// Syntax:   Tensor Gs = priv_select_generators(G, columns);
// See also: zonotope.h

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"

#include <functional>
#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Refusals ----------------------------------------------------------------------------------------

/// Why an input is refused, if it is: the identifier and the text, for CORA_CHECK in the entry.
using Refusal = std::optional<std::pair<Error::Id, std::string>>;

// Helpers -----------------------------------------------------------------------------------------

/// The given columns of G (..., n, m), in that order; none is allowed.
Tensor priv_select_generators(const Tensor &G, const std::vector<int64_t> &columns);

/// Why is_intersecting refuses a call, if it does: a type other than "exact" and "approx", or a
/// batch (a linear program per pair); c and other are the centers of the two sets.
Refusal priv_intersecting_refusal(const std::string &type, const Tensor &c, const Tensor &other);

/// The same for points p (n, N) of a set of dimension n: p must be a matrix of that many rows.
Refusal priv_intersecting_points_refusal(const std::string &type, const Tensor &c,
                                         const Tensor &p);

/// The generators (..., r, keep + r) after Girard's reduction of G (..., r, m): the keep generators
/// of the largest score ||g||_1 - ||g||_inf, then a box generator per row for the others.
Tensor priv_girard_generators(const Tensor &G, int64_t keep);

/// The shape of the batch of `like` followed by (rows, cols).
Shape priv_batch_shape(const Tensor &like, int64_t rows, int64_t cols);

/// One output of a quadratic map: its center (..., 1, 1) and a row of generators (..., 1, width).
using QuadRow = std::pair<Tensor, Tensor>;

/// The output of the single map from M = [c G]' Q [c G] (..., N+1, N+1): the center is M(0,0) plus
/// the halved squares, which are the first N generators, then the strictly lower triangle of M + M'
/// in the order of MATLAB's column-major vector.
QuadRow priv_quad_map_single_row(const Tensor &M, int64_t N);

/// The zonotope of CORA's 'linprog' union: the iterative enclosure `iterative` with unit generators
/// scaled and the center moved, by a linear program, to the least that still cover every set along
/// the facet normals of the enclosure; support(A) is the largest support value of the sets along
/// each row of A (k, n), a column (k, 1).
Zonotope priv_union_linprog(const Zonotope &iterative,
                            const std::function<Tensor(const Tensor &)> &support);

/// The zonotope of a point p (..., n, 1), as CORA's zonotope(p): center p and no generator.
Zonotope priv_zonotope_of_point(const Tensor &p);

/// Whether the generators of Z have at most one entry beyond tol each (CORA's
/// representsa_(Z, 'interval', tol)): Z is a box; of a batch, every member.
bool priv_is_box(const Zonotope &Z, double tol);

/// What is wrong with the matrices Q of a quadratic map of sets of dimension n (CORA's quadMap
/// checks), or an empty text if nothing.
std::string priv_quad_map_problem(const std::vector<Tensor> &Q, int64_t n);

/// What is wrong with the matrices T[i][k] and the index lists of a cubic map of sets of dimension
/// n (CORA's cubMap checks), or an empty text if nothing.
std::string priv_cubic_map_problem(const std::vector<std::vector<Tensor>> &T,
                                   const std::vector<std::vector<int64_t>> &ind, int64_t n);

/// Why the strips |C x - y| <= phi (C (k, n), phi and y (k, 1)) of a set of dimension n, a gain
/// (n, k) or the matrices of an observer are refused by intersect_strip, if they are.
Refusal priv_strip_refusal(int64_t n, const Tensor &C, const Tensor &phi, const Tensor &y);
Refusal priv_gain_refusal(int64_t n, const Tensor &C, const Tensor &lambda);
Refusal priv_observer_refusal(int64_t n, const Zonotope::StripObserver &aux);

/// The n-subsets of {0, ..., total - 1} in lexicographic order, produced in blocks (CORA's
/// batchCombinator): next(count) returns up to `count` subsets as one flat list, empty when done.
class Combinations {
  public:
    Combinations(int64_t total, int64_t n);
    std::vector<int64_t> next(int64_t count);

  private:
    int64_t total_, n_;
    std::vector<int64_t> current_;
    bool done_ = false;
};

/// The matrices of the columns G(:, members(b, :)) for every row b of members (B, n) (indices as
/// values of G's dtype): G (..., r, m) gives (..., B, r, n).
Tensor priv_column_subsets(const Tensor &G, const Tensor &members);

/// |det| of the matrices P (..., n, n), by the diagonal of R of a QR factorization: (..., 1, 1).
Tensor priv_abs_det(const Tensor &P);

/// A zonotope on the host: the center (n) and the generators (n x m, row-major).
struct HostZonotope {
    int64_t n = 0, m = 0;
    std::vector<double> c, G;
};

/// The values of a single (unbatched) zonotope on the host.
HostZonotope priv_host_zonotope(const Zonotope &Z);

/// The norm of a point w.r.t. the generators of a zonotope and the coefficients that reach it.
struct ZonotopeNorm {
    double value;                  ///< s, or infinity if there is no beta
    std::vector<double> minimizer; ///< beta (m values); empty if the value is infinite
};

/// The norm of the point p (n values) w.r.t. the generators of Z (its center is ignored):
/// min s such that G beta = p, |beta|_inf <= s, by a linear program; infinity if there is none
/// (CORA's zonotopeNorm).
ZonotopeNorm priv_zonotope_norm_minimizer(const HostZonotope &Z, const std::vector<double> &p);

/// The value of priv_zonotope_norm_minimizer.
double priv_zonotope_norm(const HostZonotope &Z, const std::vector<double> &p);

/// Whether the zonotope `inner` lies in `outer` (plus tol in the norm of outer), by the
/// depth-first search over the vertices of `inner` (CORA's venum search, Kulmburg et al. 2024).
bool priv_containment_vertex_enumeration(const HostZonotope &inner, const HostZonotope &outer,
                                         double tol);

/// The scaling of the sufficient test of Sadraddini and Tedrake (CDC 2019, Thm. 3): `inner` lies
/// in the scaling of `outer` about its center if that is at most 1; by a linear program.
double priv_containment_sadraddini_tedrake(const HostZonotope &inner, const HostZonotope &outer);

/// The two halves of Z along generator j (centers c -+ g_j/2, that generator halved): lower first
/// (CORA's aux_splitOneDim). A batch splits member by member.
std::pair<Zonotope, Zonotope> priv_split_generator(const Zonotope &Z, int64_t j);

/// Why tensor_multiplication refuses Z, M and list, if it does: the center is not the origin, M is
/// not a third-order tensor (n, n, n, n), or a list is not consistent.
Refusal priv_tensor_multiplication_refusal(const Zonotope &Z, const Tensor &M,
                                           const Zonotope::TensorLists &list);

/// The zonotope (0, H) of CORA's tensorMultiplication for a checked input, batch-aware.
Zonotope priv_tensor_multiplication(const Zonotope &Z, const Tensor &M,
                                    const Zonotope::TensorLists &list);

/// The unit normals of the facets of a full-dimensional zonotope as rows (k, n), in CORA's order
/// (the first half of polytope(Z).A); a single zonotope without zero generators.
Tensor priv_zonotope_facet_normals(const Zonotope &Z);

/// The halfspaces C x <= d (C (k, n), d (k, 1)) whose intersection is the single zonotope Z,
/// including a degenerate one (CORA's exact polytope(Z), without the polytope class).
std::pair<Tensor, Tensor> priv_zonotope_halfspaces(const Zonotope &Z);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
