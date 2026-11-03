// test_tensor_overrides - a backend's native version of a derived operation agrees with the default
//
// The derived and fused operations with a hook in Tensor::Impl have a portable default made of
// core operations (global/tensor/derived.h). A backend may replace one by its own (eigen: expm and
// the fused primitives; torch: expm, sort, diag, the norms and the interval products). On every
// backend, the operation and its default give the same result within the agreement of ops.def.

#include "cora/global/tensor/tensor.h"
#include "global/oracle/oracle_testing.h"
#include "global/tensor/derived.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;

namespace {

using Norm = Tensor::Norm;

/// a and b agree within rel times the largest entry of a (0: bit for bit).
void aux_agree(const std::string &what, const Tensor &a, const Tensor &b, double rel) {
    if (a.shape() != b.shape()) {
        check(false, what + ": the shapes differ");
        return;
    }
    if (rel == 0) {
        check(test::oracle::same_bits(what, a.to("cpu"), b.to("cpu")), what + ": the same bits");
        return;
    }
    const std::vector<double> x = a.data(), y = b.data();
    double largest = 0, diff = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        largest = std::max(largest, std::fabs(x[i]));
        diff = std::max(diff, std::fabs(x[i] - y[i]));
    }
    check(diff <= rel * largest,
          what + ": within " + std::to_string(rel) + " of the largest entry");
}

/// The tolerance of a sum or product of n terms: n u, with room for the order.
double aux_sum_bound(int64_t n) { return 4.0 * n * std::numeric_limits<double>::epsilon(); }

void aux_overrides(const std::string &b) {
    test::oracle::Random rnd(23);
    const Tensor A = rnd.matrix(6, 6) * 0.4, G = rnd.matrix(8, 6 * 11).reshape({8, 6, 11});
    const Tensor x = rnd.matrix(3, 9).round();
    // expm, where the backend has one (lean has none)
    if (b != "lean") aux_agree(b + " expm", A.expm(), derived::expm(A), 1e-13);
    aux_agree(b + " sort", x.sort(-1, true), derived::sort(x, -1, true), 0);
    const auto [v, i] = x.topk(4);
    const auto [dv, di] = derived::topk(x, 4, -1, true);
    aux_agree(b + " topk values", v, dv, 0);
    aux_agree(b + " topk indices", i, di, 0);
    const Tensor B = rnd.matrix(2, 3);
    aux_agree(b + " kron", A.kron(B), derived::kron(A, B), 0);
    for (const Norm ord : {Norm::one, Norm::inf, Norm::fro, Norm::two})
        aux_agree(b + " matrix_norm", G.matrix_norm(ord), derived::matrix_norm(G, ord),
                  aux_sum_bound(11));
    for (const Norm ord : {Norm::one, Norm::inf, Norm::two})
        aux_agree(b + " vector_norm", G.vector_norm(ord, -2),
                  derived::vector_norm(G, ord, -2, false), aux_sum_bound(6));
    const Tensor c = rnd.vector(6);
    aux_agree(b + " diag", c.diag(), derived::diag(c), 0);

    // the fused primitives
    const auto [Mc, MG] = Tensor::zonotope_linear_map(A, c, G);
    const auto [ref_c, ref_g] = derived::zonotope_linear_map(A, c, G);
    aux_agree(b + " zonotope_linear_map c", Mc, ref_c, aux_sum_bound(6));
    aux_agree(b + " zonotope_linear_map G", MG, ref_g, aux_sum_bound(6));
    const Tensor lo = rnd.matrix(6, 6), hi = lo + rnd.matrix(6, 6).abs();
    const auto [Tc, TG] = Tensor::zonotope_interval_map(lo, hi, c, G);
    const auto [ref_tc, ref_tg] = derived::zonotope_interval_map(lo, hi, c, G);
    aux_agree(b + " zonotope_interval_map c", Tc, ref_tc, aux_sum_bound(6));
    aux_agree(b + " zonotope_interval_map G", TG, ref_tg, aux_sum_bound(12));
    const Tensor cb = c.broadcast_to({8, 6, 1}) + G.sum(-1, true); // a center for every member
    const auto [mean_c, mean_g] = Tensor::zonotope_linear_combination(cb, G, cb * 0.5, G * 3.0);
    const auto [ref_mc, ref_mg] = derived::zonotope_linear_combination(cb, G, cb * 0.5, G * 3.0);
    aux_agree(b + " zonotope_linear_combination c", mean_c, ref_mc, 0);
    aux_agree(b + " zonotope_linear_combination G", mean_g, ref_mg, 0);
    const auto [p, q] = Tensor::interval_matmul(lo, hi, A);
    const auto [dp, dq] = derived::interval_matmul(lo, hi, A);
    aux_agree(b + " interval_matmul inf", p, dp, aux_sum_bound(6));
    aux_agree(b + " interval_matmul sup", q, dq, aux_sum_bound(6));
    const auto [r, s] = Tensor::matmul_interval(A, lo, hi);
    const auto [dr, ds] = derived::matmul_interval(A, lo, hi);
    aux_agree(b + " matmul_interval inf", r, dr, aux_sum_bound(6));
    aux_agree(b + " matmul_interval sup", s, ds, aux_sum_bound(6));
    const auto [kept, rest] = G.girard_split(4);
    const auto [dkept, drest] = derived::girard_split(G, 4);
    aux_agree(b + " girard_split kept", kept, dkept, 0);
    aux_agree(b + " girard_split rest", rest, drest, 0);
    aux_agree(b + " abs_sum", G.abs_sum(), derived::abs_sum(G), aux_sum_bound(11));
}

} // namespace

int main() {
    std::vector<std::string> all = test::backends();
    all.push_back("lean");
    for (const std::string &b : all) {
        const BackendScope scope(b);
        aux_overrides(b);
    }
    return test::finish("tensor overrides");
}
