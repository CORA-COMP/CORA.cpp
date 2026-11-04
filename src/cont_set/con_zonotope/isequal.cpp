// isequal - whether a constrained zonotope equals a set, as CORA's conZonotope.isequal
//
// Syntax:   bool res = cz.isequal(S);   bool res = cz.isequal(S, tol);
// Inputs:   S - a constrained zonotope, a zonotope or a box;  tol - what counts as equal (as
//           within_tol); default: the repository tolerance (CORA's own is 1e-12)
// Outputs:  res - two empty sets are equal and an empty and a nonempty one are not; two points
//           are equal if they agree; two sets are equal if the centers agree and the columns of
//           [G; A] and of [A b] agree up to their order, else if each contains the vertex
//           candidates of the other (CORA compares the vertices; the candidates are points of the
//           set that include its vertices, so the answer is the same and no vertex list is needed).
//           A zonotope or a box equals a set without constraints that is that zonotope. A
//           different dimension is false. Not a batch
// See also: contains, Zonotope::isequal
//
// CORA: contSet/@conZonotope/isequal.m

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Comparisons -------------------------------------------------------------------------------------

/// Whether the columns of M1 and M2 (r x k each) agree up to order: each column of M1 takes the
/// first unused column of M2 that equals it (CORA's compareMatrices).
bool aux_same_columns(const Tensor &M1, const Tensor &M2, double tol) {
    if (M1.shape() != M2.shape()) return false;
    const int64_t rows = M1.shape()[0], cols = M1.shape()[1];
    const std::vector<double> a = M1.data(), b = M2.data();
    std::vector<bool> used(std::size_t(cols), false);
    for (int64_t i = 0; i < cols; ++i) {
        bool found = false;
        for (int64_t j = 0; j < cols && !found; ++j) {
            bool equal = !used[j];
            for (int64_t r = 0; equal && r < rows; ++r)
                equal = within_tol(a[r * cols + i], b[r * cols + j], tol);
            if (equal) found = used[j] = true;
        }
        if (!found) return false;
    }
    return true;
}

/// Whether all points of P (n, K) are in the set up to tol; the empty list is.
bool aux_holds(const ConZonotope &cz, const Tensor &P, double tol) {
    return P.shape().back() == 0 || cz.contains(P, "exact", tol);
}

/// Whether two single nonempty sets of one dimension are equal.
bool aux_equal(const ConZonotope &cz, const ConZonotope &S, double tol) {
    const Interval box = cz.interval(), box_S = S.interval();
    const bool point = box.representsa("point", tol), point_S = box_S.representsa("point", tol);
    if (point || point_S)
        return point && point_S && within_tol(box.inf(), box_S.inf(), tol).min().item() == 1.0;
    if (within_tol(cz.c(), S.c(), tol).min().item() == 1.0 &&
        aux_same_columns(Tensor::concat({cz.generators(), cz.A()}, -2),
                         Tensor::concat({S.generators(), S.A()}, -2), tol) &&
        aux_same_columns(Tensor::concat({cz.A(), cz.b()}, -1), Tensor::concat({S.A(), S.b()}, -1),
                         tol))
        return true;
    return aux_holds(S, priv_vertex_candidates(cz), tol) &&
           aux_holds(cz, priv_vertex_candidates(S), tol);
}

// The answer --------------------------------------------------------------------------------------

/// Why two sets are refused: a batch (a linear program per set).
detail::Refusal aux_refusal(const ConZonotope &cz, const ConZonotope &S) {
    if (cz.c().ndim() == 2 && S.c().ndim() == 2) return std::nullopt;
    return std::pair{Error::not_supported,
                     std::string("isequal: it supports single sets, not a batch")};
}

/// The answer for two checked sets: a different dimension is false, empty sets are equal only to
/// each other.
bool aux_isequal(const ConZonotope &cz, const ConZonotope &S, double tol) {
    if (S.dim() != cz.dim()) return false;
    const bool empty = priv_is_empty(priv_host_con_zonotope(cz)),
               empty_S = priv_is_empty(priv_host_con_zonotope(S));
    if (empty || empty_S) return empty && empty_S;
    return aux_equal(cz, S, tol);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ConZonotope::isequal(const ConZonotope &S, Tol tol) const {
    detail::Entry entry;
    const double tol_value = resolve_tol(entry, tol, c_.dtype(), "isequal");
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_isequal(*this, S, tol_value);
}

bool ConZonotope::isequal(const Zonotope &Z, Tol tol) const {
    detail::Entry entry;
    const double tol_value = resolve_tol(entry, tol, c_.dtype(), "isequal");
    CORA_CHECK(c_.ndim() == 2 && Z.center().ndim() == 2, Error::not_supported,
               [] { return std::string("isequal: it supports single sets, not a batch"); });
    // a zonotope equals a set that has no constraints and is that zonotope
    return Z.dim() == dim() && !priv_is_empty(priv_host_con_zonotope(*this)) && A_.numel() == 0 &&
           Zonotope(c_, G_).isequal(Z, tol_value);
}

bool ConZonotope::isequal(const Interval &I, Tol tol) const {
    detail::Entry entry;
    // an empty box is equal to an empty set only
    if (I.is_empty()) return priv_is_empty(priv_host_con_zonotope(*this));
    return isequal(I.zonotope(), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
