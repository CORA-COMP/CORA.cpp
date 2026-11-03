// derived - the derived Tensor operations, made of core operations: one code path for every backend
//
// The operations with a virtual hook in Tensor::Impl (expm, sort, topk, kron, the norms, diag)
// have their defaults here, which a backend may replace by a native version that agrees with
// them; the others exist only here. Selections that a result depends on piecewise (sort orders,
// the signs of factors) are taken on detached values, so gradients flow through the values only.
//
// See also: derived.h, tensor.cpp, fused.cpp, docs/tensor.md

#include "global/tensor/derived.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Norm = Tensor::Norm;

/// The axes of t but its last `drop`, followed by `append`.
Shape aux_batch_with(const Tensor &t, int drop, std::initializer_list<int64_t> append) {
    Shape s;
    for (int64_t d = 0; d + drop < t.ndim(); ++d) s.push_back(t.shape()[d]);
    for (const int64_t a : append) s.push_back(a);
    return s;
}

/// 1 where x < 0, else -1 times that plus one: the signs -1 and 1 that make x nonnegative.
Tensor aux_flip_signs(const Tensor &x) {
    return x.scalar_op(Tensor::Binary::less, 0.0) * -2.0 + 1.0;
}

/// Columns of U (and V) flipped so the entry of largest magnitude of each column of U is
/// positive: the sign convention that makes svd and eigh unique.
std::pair<Tensor, Tensor> aux_fix_columns(const Tensor &U, const Tensor &V) {
    const Tensor fixed = U.detach();
    const Tensor at = fixed.abs().argmax(-2, true);
    const Tensor s = aux_flip_signs(fixed.take_along_axis(at, -2));
    return {U.multiply(s), V.multiply(s)};
}

void aux_require_matrix(const Tensor &t, const char *op) {
    if (t.ndim() < 2)
        CORA_FAIL(Error::dimension_mismatch,
                  std::string("CoraTensor::") + op + ": needs a matrix (..., n, m)");
}

/// The Pade 13 coefficients of Higham's scaling and squaring (SIAM J. Matrix Anal. Appl. 2005).
constexpr double kPade[14] = {64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
                              1187353796428800.0,  129060195264000.0,   10559470521600.0,
                              670442572800.0,      33522128640.0,       1323241920.0,
                              40840800.0,          960960.0,            16380.0,
                              182.0,               1.0};
constexpr double kTheta13 = 5.371920351148152;

} // namespace


// ===========================================  MAIN  =========================================== //

// Defaults of the Virtual Hooks -------------------------------------------------------------------

Tensor::Ptr Tensor::Impl::expm() const { return derived::expm(Tensor(self())).ptr(); }
Tensor::Ptr Tensor::Impl::sort(int axis, bool descending) const {
    return derived::sort(Tensor(self()), axis, descending).ptr();
}
std::pair<Tensor::Ptr, Tensor::Ptr> Tensor::Impl::topk(int64_t k, int axis, bool largest) const {
    const auto [values, indices] = derived::topk(Tensor(self()), k, axis, largest);
    return {values.ptr(), indices.ptr()};
}
Tensor::Ptr Tensor::Impl::kron(const Impl &o) const {
    return derived::kron(Tensor(self()), Tensor(o.self())).ptr();
}
Tensor::Ptr Tensor::Impl::matrix_norm(Norm ord) const {
    return derived::matrix_norm(Tensor(self()), ord).ptr();
}
Tensor::Ptr Tensor::Impl::vector_norm(Norm ord, int axis, bool keepdims) const {
    return derived::vector_norm(Tensor(self()), ord, axis, keepdims).ptr();
}
Tensor::Ptr Tensor::Impl::diag() const { return derived::diag(Tensor(self())).ptr(); }

// Scaling and squaring with the Pade approximant of degree 13; one scaling for the whole batch.
Tensor derived::expm(const Tensor &A) {
    aux_require_matrix(A, "expm");
    const int64_t n = A.size(-1);
    if (A.size(-2) != n) CORA_FAIL(Error::dimension_mismatch, "CoraTensor::expm: needs n x n");
    const std::vector<double> norms = A.detach().abs().sum(-2).max(-1).data();
    const double norm = norms.empty() ? 0.0 : *std::max_element(norms.begin(), norms.end());
    const int s = norm > kTheta13 ? static_cast<int>(std::ceil(std::log2(norm / kTheta13))) : 0;
    const Tensor X = A * std::ldexp(1.0, -s), I = A.eye_like();
    const Tensor X2 = X.matmul(X), X4 = X2.matmul(X2), X6 = X4.matmul(X2);
    const Tensor u_high = X6 * kPade[13] + X4 * kPade[11] + X2 * kPade[9];
    const Tensor U = X.matmul(X6.matmul(u_high) + X6 * kPade[7] + X4 * kPade[5] + X2 * kPade[3] +
                              I * kPade[1]);
    const Tensor v_high = X6 * kPade[12] + X4 * kPade[10] + X2 * kPade[8];
    const Tensor V = X6.matmul(v_high) + X6 * kPade[6] + X4 * kPade[4] + X2 * kPade[2] +
                     I * kPade[0];
    Tensor R = (V - U).solve(V + U);
    for (int k = 0; k < s; ++k) R = R.matmul(R); // undo the scaling by squaring
    return R;
}

// The order on detached values; the values themselves keep their gradients.
Tensor derived::sort(const Tensor &x, int axis, bool descending) {
    return x.take_along_axis(x.detach().argsort(axis, descending), axis);
}

std::pair<Tensor, Tensor> derived::topk(const Tensor &x, int64_t k, int axis, bool largest) {
    const Tensor idx = x.detach().argsort(axis, largest).slice(axis, 0, k);
    return {x.take_along_axis(idx, axis), idx};
}

// Each entry is one product a_ij b_kl, at (i p + k, j q + l).
Tensor derived::kron(const Tensor &A, const Tensor &B) {
    aux_require_matrix(A, "kron");
    aux_require_matrix(B, "kron");
    const int64_t n = A.size(-2), m = A.size(-1), p = B.size(-2), q = B.size(-1);
    const Tensor a = A.reshape(aux_batch_with(A, 2, {n, 1, m, 1}));
    const Tensor b = B.reshape(aux_batch_with(B, 2, {1, p, 1, q}));
    const Tensor big = a.multiply(b);
    return big.reshape(aux_batch_with(big, 4, {n * p, m * q}));
}

// 1: the largest column sum of |A|; inf: the largest row sum; 2: the largest singular value.
Tensor derived::matrix_norm(const Tensor &A, Norm ord) {
    aux_require_matrix(A, "matrix_norm");
    switch (ord) {
    case Norm::one: return A.abs().sum(-2, true).max(-1, true);
    case Norm::inf: return A.abs().sum(-1, true).max(-2, true);
    case Norm::fro: return A.square().sum(-1, true).sum(-2, true).sqrt();
    case Norm::two: return A.svd()[1].max(-2, true);
    }
    CORA_FAIL(Error::wrong_value, "CoraTensor::matrix_norm: unknown norm");
}

Tensor derived::vector_norm(const Tensor &x, Norm ord, int axis, bool keepdims) {
    switch (ord) {
    case Norm::one: return x.abs().sum(axis, keepdims);
    case Norm::inf: return x.abs().max(axis, keepdims);
    case Norm::two:
    case Norm::fro: return x.square().sum(axis, keepdims).sqrt();
    }
    CORA_FAIL(Error::wrong_value, "CoraTensor::vector_norm: unknown norm");
}

// The column's entries where row and column agree, zeros elsewhere.
Tensor derived::diag(const Tensor &v) {
    const int64_t n = v.size(-2);
    return Tensor::where(v.eye_like(n), v, v.zeros_like({1}));
}

// Elementwise -------------------------------------------------------------------------------------

Tensor Tensor::square() const { return multiply(*this); }

Tensor Tensor::sigmoid() const {
    return negative().exp().scalar_op(Binary::add, 1.0).scalar_op(Binary::divide, 1.0, true);
}

Tensor Tensor::clip(double low, double high) const { return maximum(low).minimum(high); }

// Reductions --------------------------------------------------------------------------------------

Tensor Tensor::mean(int axis, bool keepdims) const {
    return sum(axis, keepdims) / static_cast<double>(size(axis));
}

// Counting the nonzero entries works for an empty axis too.
Tensor Tensor::any(int axis, bool keepdims) const {
    const Tensor x = detail::without_gradient(TensorOp::any, *this);
    return x.count_nonzero(axis, keepdims).scalar_op(Binary::greater, 0.0);
}

Tensor Tensor::all(int axis, bool keepdims) const {
    const Tensor x = detail::without_gradient(TensorOp::all, *this);
    const double len = static_cast<double>(size(axis));
    return x.count_nonzero(axis, keepdims).scalar_op(Binary::equal, len);
}

Tensor Tensor::count_nonzero(int axis, bool keepdims) const {
    const Tensor x = detail::without_gradient(TensorOp::count_nonzero, *this);
    return x.scalar_op(Binary::not_equal, 0.0).sum(axis, keepdims);
}

// Shifted by the (detached) maximum, which cancels in the quotient.
Tensor Tensor::softmax(int axis) const {
    const Tensor e = (*this - detach().max(axis, true)).exp();
    return e.divide(e.sum(axis, true));
}

Tensor Tensor::sum() const { return reshape({numel()}).sum(0).reshape({1, 1}); }
Tensor Tensor::max() const { return reshape({numel()}).max(0).reshape({1, 1}); }
Tensor Tensor::min() const { return reshape({numel()}).min(0).reshape({1, 1}); }

// Shape and Selection -----------------------------------------------------------------------------

// Each part with a new axis of size 1, joined along it.
Tensor Tensor::stack(const std::vector<Tensor> &parts, int axis) {
    if (parts.empty()) CORA_FAIL(Error::wrong_value, "CoraTensor::stack: no tensors given");
    std::vector<Tensor> lifted;
    for (const Tensor &p : parts) {
        if (p.shape() != parts[0].shape())
            CORA_FAIL(Error::dimension_mismatch, "CoraTensor::stack: the tensors differ in shape");
        lifted.push_back(p.expand_dims(axis));
    }
    return concat(lifted, axis);
}

Tensor Tensor::matrix_transpose() const {
    aux_require_matrix(*this, "matrix_transpose");
    std::vector<int> axes;
    for (int d = 0; d < ndim(); ++d) axes.push_back(d);
    std::swap(axes[ndim() - 1], axes[ndim() - 2]);
    return permute_dims(axes);
}

// Views through reshape.
Tensor Tensor::expand_dims(int axis) const {
    const int a = detail::axis_of(axis, ndim() + 1, "expand_dims");
    Shape s;
    for (int d = 0; d <= ndim(); ++d) s.push_back(d == a ? 1 : shape()[d < a ? d : d - 1]);
    return reshape(s);
}

Tensor Tensor::squeeze(int axis) const {
    const int a = detail::axis_of(axis, ndim(), "squeeze");
    if (shape()[a] != 1) CORA_FAIL(Error::wrong_value, "CoraTensor::squeeze: the axis is not 1");
    Shape s;
    for (int d = 0; d < ndim(); ++d)
        if (d != a) s.push_back(shape()[d]);
    return reshape(s);
}

// The indices n-1, ..., 0.
Tensor Tensor::flip(int axis) const {
    const int64_t n = size(axis);
    return take(arange_like(n).scalar_op(Binary::subtract, double(n - 1), true), axis);
}

// x of shape (s0, s1, ...) as (1, s0, 1, s1, ...), broadcast to (r0, s0, r1, s1, ...).
Tensor Tensor::tile(const std::vector<int64_t> &reps) const {
    if (static_cast<int64_t>(reps.size()) != ndim())
        CORA_FAIL(Error::wrong_value, "CoraTensor::tile: give one repetition per axis");
    if (ndim() * 2 > Shape::kMaxRank)
        CORA_FAIL(Error::not_supported, "CoraTensor::tile: too many axes");
    Shape lifted, wide, out;
    for (int d = 0; d < ndim(); ++d) {
        lifted.push_back(1);
        lifted.push_back(shape()[d]);
        wide.push_back(reps[d]);
        wide.push_back(shape()[d]);
        out.push_back(reps[d] * shape()[d]);
    }
    return reshape(lifted).broadcast_to(wide).reshape(out);
}

// Entry (i, i) of an (n, m) matrix is entry i (m + 1) of its rows laid end to end.
Tensor Tensor::diagonal() const {
    aux_require_matrix(*this, "diagonal");
    const int64_t n = size(-2), m = size(-1), k = std::min(n, m);
    const Tensor flat = reshape(aux_batch_with(*this, 2, {n * m}));
    return flat.slice(-1, 0, k == 0 ? 0 : (k - 1) * (m + 1) + 1, m + 1)
        .reshape(aux_batch_with(*this, 2, {k, 1}));
}

// Kept where j - i <= k.
Tensor Tensor::tril(int64_t k) const {
    const int64_t n = size(-2), m = size(-1);
    const Tensor i = arange_like(n).reshape({n, 1}), j = arange_like(m).reshape({1, m});
    const Tensor keep = (j - i).scalar_op(Binary::less_equal, double(k));
    return where(keep, *this, zeros_like({1}));
}

Tensor Tensor::triu(int64_t k) const {
    const int64_t n = size(-2), m = size(-1);
    const Tensor i = arange_like(n).reshape({n, 1}), j = arange_like(m).reshape({1, m});
    const Tensor keep = (j - i).scalar_op(Binary::greater_equal, double(k));
    return where(keep, *this, zeros_like({1}));
}

// Each block's rows padded with zeros left and right, the rows stacked.
Tensor Tensor::block_diag(const std::vector<Tensor> &blocks) {
    if (blocks.empty()) CORA_FAIL(Error::wrong_value, "CoraTensor::block_diag: no blocks given");
    int64_t total = 0;
    for (const Tensor &b : blocks) {
        aux_require_matrix(b, "block_diag");
        total += b.size(-1);
    }
    std::vector<Tensor> rows;
    int64_t before = 0;
    for (const Tensor &b : blocks) {
        const int64_t n = b.size(-2), m = b.size(-1);
        std::vector<Tensor> parts;
        if (before > 0) parts.push_back(b.zeros_like(aux_batch_with(b, 2, {n, before})));
        parts.push_back(b);
        if (total - before - m > 0)
            parts.push_back(b.zeros_like(aux_batch_with(b, 2, {n, total - before - m})));
        rows.push_back(concat(parts, -1));
        before += m;
    }
    return concat(rows, -2);
}

// Sorting -----------------------------------------------------------------------------------------

std::vector<int64_t> Tensor::nonzero() const {
    const std::vector<double> values = detail::without_gradient(TensorOp::nonzero, *this).data();
    std::vector<int64_t> at;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (values[i] != 0) at.push_back(static_cast<int64_t>(i));
    return at;
}

// Linear Algebra ----------------------------------------------------------------------------------

Tensor Tensor::inv() const { return solve(eye_like(size(-1))); }

// With A = Q R (R square for full column rank): R x = Q^T B.
Tensor Tensor::lstsq(const Tensor &B) const {
    const auto [Q, R] = qr();
    return R.solve(Q.matrix_transpose().matmul(B));
}

// V diag(1/s) U^T over the singular values above max(n, m) eps max(s).
Tensor Tensor::pinv() const {
    const auto [U, S, V] = svd();
    const Tensor bound =
        S.detach().max(-2, true) * (machine_eps(dtype()) * double(std::max(size(-2), size(-1))));
    const Tensor keep = S.detach().greater(bound);
    // the reciprocal of a kept value only, so that no infinity meets a gradient
    const Tensor inv_s = where(keep, 1.0 / where(keep, S, S.ones_like({1})), S.zeros_like({1}));
    return V.multiply(inv_s.matrix_transpose()).matmul(U.matrix_transpose());
}

int64_t Tensor::matrix_rank(double tol) const {
    const Tensor x = detail::without_gradient(TensorOp::matrix_rank, *this);
    if (x.ndim() != 2) CORA_FAIL(Error::wrong_value, "CoraTensor::matrix_rank: one matrix (n, m)");
    const std::vector<double> s = x.svd()[1].data();
    if (s.empty()) return 0;
    const double bound =
        tol >= 0 ? tol : double(std::max(size(-2), size(-1))) * machine_eps(dtype()) * s[0];
    return std::count_if(s.begin(), s.end(), [&](double v) { return v > bound; });
}

// The backend's factors, with the signs that make diag(R) nonnegative.
std::pair<Tensor, Tensor> Tensor::qr() const {
    aux_require_matrix(*this, "qr");
    const auto [q, r] = impl_->qr();
    const Tensor Q(q), R(r);
    const Tensor s = aux_flip_signs(R.detach().diagonal()); // (..., k, 1)
    return {Q.multiply(s.matrix_transpose()), R.multiply(s)};
}

std::array<Tensor, 3> Tensor::svd() const {
    aux_require_matrix(*this, "svd");
    const auto [u, s, v] = impl_->svd();
    const auto [U, V] = aux_fix_columns(Tensor(u), Tensor(v));
    return {U, Tensor(s), V};
}

std::pair<Tensor, Tensor> Tensor::eigh() const {
    aux_require_matrix(*this, "eigh");
    const auto [w, v] = impl_->eigh();
    const Tensor V(v);
    return {Tensor(w), aux_fix_columns(V, V).first};
}

Tensor Tensor::trace() const { return diagonal().sum(-2, true); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
