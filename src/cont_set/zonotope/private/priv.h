// priv - the private functions of zonotope, as CORA's @zonotope/private
//
// Not for callers of the class: the operations that mix zonotopes with other sets use them.
//
// Syntax:   Tensor Gs = priv_select_generators(G, columns);
// See also: zonotope.h

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Why an input is refused, if it is: the identifier and the text, for CORA_CHECK in the entry.
using Refusal = std::optional<std::pair<Error::Id, std::string>>;

/// The given columns of G (..., n, m), in that order; none is allowed.
Tensor priv_select_generators(const Tensor &G, const std::vector<int64_t> &columns);

/// The zonotope of a point p (..., n, 1), as CORA's zonotope(p): center p and no generator.
Zonotope priv_zonotope_of_point(const Tensor &p);

/// Whether the generators of Z have at most one entry beyond tol each (CORA's
/// representsa_(Z, 'interval', tol)): Z is a box; of a batch, every member.
bool priv_is_box(const Zonotope &Z, double tol);

/// What is wrong with the matrices Q of a quadratic map of sets of dimension n (CORA's quadMap
/// checks), or an empty text if nothing.
std::string priv_quad_map_problem(const std::vector<Tensor> &Q, int64_t n);
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
    double value;                 ///< s, or infinity if there is no beta
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
