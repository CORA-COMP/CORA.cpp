// fused - the fused Tensor primitives of the hot spots, each with one specification
//
// A fused primitive is an operation like any other: its default here, made of core operations,
// is the specification, and on the lean backend it rounds and orders exactly as written (sums
// nested to the right). A backend implements the same specification natively (eigen.cpp,
// torch.cpp), within the agreement of ops.def.
//
// See also: derived.h, derived.cpp, ops.def
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "global/tensor/derived.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void aux_require_matrix(const Tensor &t, const char *op) {
    if (t.ndim() < 2)
        CORA_FAIL(Error::dimension_mismatch,
                  std::string("CoraTensor::") + op + ": needs matrices (..., n, m)");
}

/// The bounds of an interval matrix must agree in shape.
void aux_require_bounds(const Tensor &inf, const Tensor &sup, const char *op) {
    detail::check_pair(inf, sup, op);
    aux_require_matrix(inf, op);
    if (inf.shape() != sup.shape())
        CORA_FAIL(Error::dimension_mismatch,
                  std::string("CoraTensor::") + op + ": the bounds differ in shape");
}

/// The batch axes (all but the last two) of matrices, broadcast as in numpy; no heap memory.
Shape aux_common_batch(std::initializer_list<const Tensor *> ts, const char *op) {
    int64_t rank = 0;
    for (const Tensor *t : ts) rank = std::max<int64_t>(rank, t->ndim() - 2);
    Shape out;
    for (int64_t i = 0; i < rank; ++i) out.push_back(1);
    for (const Tensor *t : ts) {
        const int64_t r = t->ndim() - 2;
        for (int64_t i = 0; i < r; ++i) {
            int64_t &o = out[rank - r + i];
            const int64_t d = t->shape()[i];
            if (d != 1 && o != 1 && d != o)
                CORA_FAIL(Error::dimension_mismatch,
                          std::string("CoraTensor::") + op + ": the batch axes do not broadcast");
            if (o == 1) o = d;
        }
    }
    return out;
}

/// t with the batch axes `batch` before its last two, broadcast where they differ.
Tensor aux_with_batch(const Tensor &t, const Shape &batch) {
    Shape target = batch;
    target.push_back(t.size(-2));
    target.push_back(t.size(-1));
    return t.shape() == target ? t : t.broadcast_to(target);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Defaults of the Virtual Hooks -------------------------------------------------------------------

std::pair<Tensor::Ptr, Tensor::Ptr> Tensor::Impl::zonotope_linear_map(const Impl &c,
                                                                      const Impl &G) const {
    const auto [Mc, MG] =
        derived::zonotope_linear_map(Tensor(self()), Tensor(c.self()), Tensor(G.self()));
    return {Mc.ptr(), MG.ptr()};
}

// The zonotope maps and the linear combination
std::pair<Tensor::Ptr, Tensor::Ptr>
Tensor::Impl::zonotope_interval_map(const Impl &sup, const Impl &c, const Impl &G) const {
    const auto [Tc, TG] = derived::zonotope_interval_map(Tensor(self()), Tensor(sup.self()),
                                                         Tensor(c.self()), Tensor(G.self()));
    return {Tc.ptr(), TG.ptr()};
}

std::pair<Tensor::Ptr, Tensor::Ptr>
Tensor::Impl::zonotope_linear_combination(const Impl &G1, const Impl &c2, const Impl &G2) const {
    const auto [c, G] = derived::zonotope_linear_combination(Tensor(self()), Tensor(G1.self()),
                                                             Tensor(c2.self()), Tensor(G2.self()));
    return {c.ptr(), G.ptr()};
}

// The interval products, both ways round
std::pair<Tensor::Ptr, Tensor::Ptr> Tensor::Impl::interval_matmul(const Impl &sup,
                                                                  const Impl &M) const {
    const auto [lo, hi] =
        derived::interval_matmul(Tensor(self()), Tensor(sup.self()), Tensor(M.self()));
    return {lo.ptr(), hi.ptr()};
}

std::pair<Tensor::Ptr, Tensor::Ptr> Tensor::Impl::matmul_interval(const Impl &inf,
                                                                  const Impl &sup) const {
    const auto [lo, hi] =
        derived::matmul_interval(Tensor(self()), Tensor(inf.self()), Tensor(sup.self()));
    return {lo.ptr(), hi.ptr()};
}

// the indices are no gradient carriers: only take() with them is differentiable
std::pair<Tensor::Ptr, Tensor::Ptr> Tensor::Impl::girard_split(int64_t keep) const {
    const auto [kept, rest] = derived::girard_split(Tensor(self()), keep);
    return {kept.ptr(), rest.ptr()};
}

Tensor::Ptr Tensor::Impl::abs_sum() const { return derived::abs_sum(Tensor(self())).ptr(); }

// The Specifications ------------------------------------------------------------------------------

std::pair<Tensor, Tensor> derived::zonotope_linear_map(const Tensor &M, const Tensor &c,
                                                       const Tensor &G) {
    return {M.matmul(c), M.matmul(G)};
}

// The center T maps the zonotope; the radius S maps the row sums of |[c G]| to box generators.
std::pair<Tensor, Tensor> derived::zonotope_interval_map(const Tensor &inf, const Tensor &sup,
                                                         const Tensor &c, const Tensor &G) {
    const Tensor T = (inf + sup) * 0.5, S = (sup - inf) * 0.5;
    const Tensor reach = c.abs() + G.abs_sum();
    return {T.matmul(c), Tensor::concat({T.matmul(G), S.matmul(reach).diag()}, -1)};
}

std::pair<Tensor, Tensor> derived::zonotope_linear_combination(const Tensor &c1, const Tensor &G1,
                                                               const Tensor &c2, const Tensor &G2) {
    const Tensor mean = (G1 + G2) * 0.5, spread = (G1 - G2) * 0.5, shift = (c1 - c2) * 0.5;
    return {(c1 + c2) * 0.5, Tensor::concat({mean, spread, shift}, -1)};
}

// Entry (i, j) sums over q the smaller (larger) of inf_iq m_qj and sup_iq m_qj, as (.., n, q, m).
std::pair<Tensor, Tensor> derived::interval_matmul(const Tensor &inf, const Tensor &sup,
                                                   const Tensor &M) {
    const Tensor a = inf.expand_dims(-1), b = sup.expand_dims(-1), m = M.expand_dims(-3);
    const Tensor pa = a.multiply(m), pb = b.multiply(m);
    return {pa.minimum(pb).sum(-2), pa.maximum(pb).sum(-2)};
}

std::pair<Tensor, Tensor> derived::matmul_interval(const Tensor &M, const Tensor &inf,
                                                   const Tensor &sup) {
    const Tensor m = M.expand_dims(-1), a = inf.expand_dims(-3), b = sup.expand_dims(-3);
    const Tensor pa = m.multiply(a), pb = m.multiply(b);
    return {pa.minimum(pb).sum(-2), pa.maximum(pb).sum(-2)};
}

// Score sum|g| - max|g| per column, then a stable sort, largest first.
std::pair<Tensor, Tensor> derived::girard_split(const Tensor &G, int64_t keep) {
    const Tensor A = G.abs();
    const Tensor order = (A.sum(-2) - A.max(-2)).argsort(-1, true).expand_dims(-2);
    return {order.slice(-1, 0, keep), order.slice(-1, keep, G.size(-1))};
}

Tensor derived::abs_sum(const Tensor &x) { return x.abs().sum(-1, true); }

// The Front End -----------------------------------------------------------------------------------

// The operands checked, then the backend's version.
std::pair<Tensor, Tensor> Tensor::zonotope_linear_map(const Tensor &M, const Tensor &c,
                                                      const Tensor &G) {
    detail::check_pair(M, c, "zonotope_linear_map");
    detail::check_pair(M, G, "zonotope_linear_map");
    const auto [Mc, MG] = M.impl().zonotope_linear_map(c.impl(), G.impl());
    return {Tensor(Mc), Tensor(MG)};
}

// The bounds agree in shape, the inner sizes of the product match.
std::pair<Tensor, Tensor> Tensor::zonotope_interval_map(const Tensor &inf, const Tensor &sup,
                                                        const Tensor &c, const Tensor &G) {
    aux_require_bounds(inf, sup, "zonotope_interval_map");
    detail::check_pair(inf, c, "zonotope_interval_map");
    detail::check_pair(inf, G, "zonotope_interval_map");
    const auto [Tc, TG] = inf.impl().zonotope_interval_map(sup.impl(), c.impl(), G.impl());
    return {Tensor(Tc), Tensor(TG)};
}

std::pair<Tensor, Tensor> Tensor::zonotope_linear_combination(const Tensor &c1, const Tensor &G1,
                                                              const Tensor &c2, const Tensor &G2) {
    for (const Tensor *t : {&G1, &c2, &G2})
        detail::check_pair(c1, *t, "zonotope_linear_combination");
    for (const Tensor *t : {&c1, &G1, &c2, &G2})
        aux_require_matrix(*t, "zonotope_linear_combination");
    const int64_t n = G1.size(-2), m = G1.size(-1);
    if (G2.size(-2) != n || G2.size(-1) != m || c1.size(-2) != n || c2.size(-2) != n ||
        c1.size(-1) != 1 || c2.size(-1) != 1)
        CORA_FAIL(Error::dimension_mismatch,
                  "CoraTensor::zonotope_linear_combination: needs centers (..., n, 1) and "
                  "generators (..., n, m) alike");
    // the shift (c1 - c2)/2 becomes a generator: all four take one batch, broadcast
    const Shape batch = aux_common_batch({&c1, &G1, &c2, &G2}, "zonotope_linear_combination");
    const Tensor a = aux_with_batch(c1, batch), A = aux_with_batch(G1, batch);
    const Tensor b = aux_with_batch(c2, batch), B = aux_with_batch(G2, batch);
    const auto [c, G] = a.impl().zonotope_linear_combination(A.impl(), b.impl(), B.impl());
    return {Tensor(c), Tensor(G)};
}

// The interval products: bounds of one shape, a real matrix M, and matching inner sizes
std::pair<Tensor, Tensor> Tensor::interval_matmul(const Tensor &inf, const Tensor &sup,
                                                  const Tensor &M) {
    aux_require_bounds(inf, sup, "interval_matmul");
    detail::check_pair(inf, M, "interval_matmul");
    aux_require_matrix(M, "interval_matmul");
    if (inf.size(-1) != M.size(-2))
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::interval_matmul: inner sizes differ");
    const auto [lo, hi] = inf.impl().interval_matmul(sup.impl(), M.impl());
    return {Tensor(lo), Tensor(hi)};
}

std::pair<Tensor, Tensor> Tensor::matmul_interval(const Tensor &M, const Tensor &inf,
                                                  const Tensor &sup) {
    aux_require_bounds(inf, sup, "matmul_interval");
    detail::check_pair(M, inf, "matmul_interval");
    aux_require_matrix(M, "matmul_interval");
    if (M.size(-1) != inf.size(-2))
        CORA_FAIL(Error::dimension_mismatch, "CoraTensor::matmul_interval: inner sizes differ");
    const auto [lo, hi] = M.impl().matmul_interval(inf.impl(), sup.impl());
    return {Tensor(lo), Tensor(hi)};
}

// Indices carry no gradient: a tensor that needs them is detached first.
std::pair<Tensor, Tensor> Tensor::girard_split(int64_t keep) const {
    aux_require_matrix(*this, "girard_split");
    if (keep < 0 || keep > size(-1))
        CORA_FAIL(Error::wrong_value, "CoraTensor::girard_split: keep " + std::to_string(keep) +
                                          " of " + std::to_string(size(-1)) + " generators");
    const Tensor G = detail::without_gradient(TensorOp::girard_split, *this);
    const auto [kept, rest] = G.impl().girard_split(keep);
    return {Tensor(kept), Tensor(rest)};
}

Tensor Tensor::abs_sum() const {
    if (ndim() < 1) CORA_FAIL(Error::dimension_mismatch, "CoraTensor::abs_sum: needs an axis");
    return Tensor(impl_->abs_sum());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
