// linear_map - the image of a polytope under a matrix, as CORA's polytope.mtimes
//
// Exact. A zero matrix gives the origin, a vertex set its mapped vertices (also in a batch), the
// whole space the whole space of the image if M has full row rank. For halfspaces: a scalar scales
// (a negative one flips the rows), an invertible square M divides the rows (A / M). Else the
// singular value decomposition M = U S V' splits the coordinates w = V'x: the ones with a singular
// value are kept (the others projected out), and the image is U_r z with z = S_r w, which is the
// rows (A V_r S_r^-1) U_r' together with the equalities U_perp' y = 0 if r < m. CORA's two branches
// (wide and tall) need full rank; this covers any rank. Not a batch except for vertices and
// invertible square matrices.
//
// Syntax:   Pres = P.linear_map(M);   Pres = M * P;
// Inputs:   M - a matrix (..., m, n)
// Outputs:  Pres - {M x | x in P}
// See also: Polytope::plus, Polytope::project
//
// CORA: contSet/@polytope/mtimes.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The images --------------------------------------------------------------------------------------

/// The rows of the halfspaces mapped by the scalar c: the offsets scale, a negative c flips A.
Polytope aux_scale(const Polytope &P, const Tensor &M) {
    const double c = M.item();
    if (c == 0.0) return Polytope::origin(P.dim());
    Polytope::Constraints C = P.constraints();
    if (c < 0) {
        C.A = -C.A;
        C.b = -C.b;
    }
    C.b = C.b * c;
    C.be = C.be * c;
    if (P.is_v_rep() && !P.is_h_rep()) return Polytope(P.V() * c);
    return {C.A, C.b, C.Ae, C.be};
}

/// The rows of A / M for an invertible square M, as MATLAB's mrdivide: (M' \ A')'.
Tensor aux_divide(const Tensor &A, const Tensor &M) {
    if (A.shape()[A.ndim() - 2] == 0) return A;
    return M.matrix_transpose().solve(A.matrix_transpose()).matrix_transpose();
}

// By the decomposition ----------------------------------------------------------------------------

/// The image by the singular value decomposition of M (m, n) with rank r.
Polytope aux_decomposed(const Polytope &P, const Tensor &M, int64_t r) {
    const int64_t m = M.shape()[0], n = M.shape()[1];
    const std::array<Tensor, 3> usv = M.svd();
    const Tensor Ur = usv[0].slice(-1, 0, r), Vr = usv[2].slice(-1, 0, r);
    // the coordinates w = V'x: V_r S_r^-1 scaled, then the ones in the kernel of M
    const Tensor scaled = Vr.divide(usv[1].slice(-2, 0, r).matrix_transpose());
    const Tensor rotation = r < n ? Tensor::concat({scaled, priv_null_space(M)}, -1) : scaled;
    const Polytope::Constraints C = P.constraints();
    Polytope::Constraints rotated{C.A.matmul(rotation), C.b, C.Ae.matmul(rotation), C.be};
    if (r < n) {
        // project the kernel coordinates out: the first r are z = S_r w
        std::vector<int64_t> keep(static_cast<std::size_t>(r));
        std::iota(keep.begin(), keep.end(), int64_t{0});
        const std::optional<HostH> z = priv_project_h(priv_host_h(rotated), keep);
        if (!z) return Polytope::empty(m);
        rotated = priv_tensors_h(C.A, *z);
    }
    // back to the space of the image: y = U_r z, and U_perp'y = 0 for the directions it misses
    const Tensor UrT = Ur.matrix_transpose();
    Tensor Ae = rotated.Ae.matmul(UrT), be = rotated.be;
    if (r < m) {
        Ae = Tensor::concat({Ae, priv_null_space(M.matrix_transpose()).matrix_transpose()}, -2);
        be = Tensor::concat({be, priv_zeros(be, m - r, 1)}, -2);
    }
    return {rotated.A.matmul(UrT), rotated.b, Ae, be};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.defined() && M.ndim() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch,
               [&] {
                   return "linear_map: the matrix needs " + std::to_string(dim()) +
                          " columns (the dimension of the set), got the shape " +
                          shape_text(M.defined() ? M.shape() : Shape{});
               });
    const int64_t m = M.shape()[M.ndim() - 2], n = dim();
    // the zero matrix maps to the origin, whatever the set
    if (M.detach().abs().max().item() <= numerics::kPolytopeZeroMatrix) return origin(m);
    const bool single = M.ndim() == 2 && (is_h_rep() ? A_ : V_).ndim() == 2;
    if (is_v_rep()) return Polytope(M.matmul(V_));
    if (single && m == 1 && n == 1) return aux_scale(*this, M);
    // invertible square matrices (a batch is assumed invertible) divide the rows
    if (m == n && (!single || M.matrix_rank() == n))
        return {aux_divide(A_, M), b_, aux_divide(Ae_, M), be_};
    CORA_CHECK(single, Error::not_supported, [] {
        return std::string("linear_map: the image of halfspaces by a non-invertible matrix "
                           "supports a single set and matrix, not a batch");
    });
    const int64_t r = M.matrix_rank();
    if (priv_is_fullspace(*this) && r == m) return inf(m);
    return aux_decomposed(*this, M, r);
}

Set Polytope::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
