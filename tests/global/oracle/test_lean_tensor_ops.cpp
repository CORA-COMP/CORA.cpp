// test_lean_tensor_ops - the lean backend's Tensor primitives against the CORALean oracle
//
// On a point interval [x, x] at dtype nearest the oracle rounds both bounds to nearest, so its
// result is the point operation in CORALean's order. The lean backend must give it bit for bit:
// elementwise arithmetic, sums and products along an axis, the matrix product, the rearrangements
// and the fused interval products (CORA's mtimes of an interval matrix), and with them the map
// M I of an interval.
//
// ORACLE: contSet/@interval/mtimes.m
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/interval.h"
#include "oracle_testing.h"

using namespace cora;
using test::oracle::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Runs one comparison; an error of the oracle fails it with the message.
template <class F>
void aux_case(const std::string &name, F f) {
    try {
        f();
    } catch (const std::exception &e) {
        test::check(false, name + ": " + e.what());
    }
}

/// The point interval [x, x] of the oracle.
lean::Interval aux_point(const Tensor &x) { return lean::Interval::from(x, x); }

/// Elementwise operations of two columns (the oracle's boxes). Not here: divide (the oracle
/// multiplies by the reciprocal) and sign (its sign of an interval is another function).
void aux_elementwise(test::oracle::Random &rnd) {
    const Tensor a = rnd.vector(6), b = rnd.vector(6);
    const lean::Interval A = aux_point(a), B = aux_point(b);
    aux_case("add", [&] { check("add", a + b, A.plus(B)); });
    aux_case("subtract", [&] { check("subtract", a - b, A.minus(B)); });
    aux_case("multiply", [&] { check("multiply", a.multiply(b), A.times(B)); });
    aux_case("maximum", [&] { check("maximum", a.maximum(b), A.max(B)); });
    aux_case("minimum", [&] { check("minimum", a.minimum(b), A.min(B)); });
    aux_case("abs", [&] { check("abs", a.abs(), A.abs()); });
    aux_case("negative", [&] { check("negative", a.negative(), A.uminus()); });
    aux_case("round", [&] { check("round", (a * 3.0).round(), aux_point(a * 3.0).round()); });
}

/// Sums along both axes (the oracle answers columns), the product along rows, and the matrix
/// product.
void aux_sums(test::oracle::Random &rnd) {
    for (const auto &[n, m] : {std::pair<int64_t, int64_t>{2, 3}, {7, 30}}) {
        const Tensor x = rnd.matrix(n, m), y = rnd.matrix(m, n), v = rnd.vector(n);
        const lean::Interval X = aux_point(x);
        aux_case("sum(-2)", [&] { check("sum(-2)", x.sum(-2, true).reshape({m, 1}), X.sum(1)); });
        aux_case("sum(-1)", [&] { check("sum(-1)", x.sum(-1, true), X.sum(2)); });
        // the product of the rows is a row in the oracle from batch 5, a column before
        aux_case("prod(-2)", [&] {
            const lean::Interval P = X.prod();
            if (P.inf().rows() == 1)
                check("prod(-2)", x.prod(-2, true), P);
            else
                test::oracle::skipped(
                    "the oracle's interval.prod of a matrix is a column (older oracle)");
        });
        aux_case("matmul", [&] { check("matmul", y.matmul(v), aux_point(v).linear_map(y)); });
    }
}

/// Rearrangements: transpose, kron, diag, tril, triu, concatenation.
void aux_rearrangements(test::oracle::Random &rnd) {
    const Tensor a = rnd.matrix(3, 4), b = rnd.matrix(2, 3), v = rnd.vector(4);
    const lean::Interval A = aux_point(a), B = aux_point(b);
    aux_case("matrix_transpose",
             [&] { check("matrix_transpose", a.matrix_transpose(), A.transpose()); });
    aux_case("kron", [&] { check("kron", a.kron(b), A.kron(B)); });
    aux_case("diag", [&] { check("diag", v.diag(), aux_point(v).diag()); });
    aux_case("tril", [&] { check("tril", a.tril(1), A.tril(1)); });
    aux_case("triu", [&] { check("triu", a.triu(2), A.triu(2)); });
    const Tensor c = rnd.matrix(3, 2);
    aux_case("concat(-1)",
             [&] { check("concat(-1)", Tensor::concat({a, c}, -1), A.horzcat(aux_point(c))); });
    const Tensor d = rnd.matrix(2, 4);
    aux_case("concat(-2)",
             [&] { check("concat(-2)", Tensor::concat({a, d}, -2), A.vertcat(aux_point(d))); });
}

/// The fused products of an interval matrix and a matrix, on either side (the oracle multiplies a
/// matrix on the left of a box only).
void aux_interval_products(test::oracle::Random &rnd) {
    for (const int64_t n : {2, 6}) {
        const Interval I = rnd.box(n * n);
        const Tensor inf = I.inf().reshape({n, n}), sup = I.sup().reshape({n, n});
        const Tensor M = rnd.matrix(n, n);
        const lean::Interval J = lean::Interval::from(inf, sup);
        const auto [lo, hi] = Tensor::interval_matmul(inf, sup, M);
        aux_case("interval_matmul",
                 [&] { check("interval_matmul", Interval(lo, hi), aux_point(M).linear_map(J)); });
        const Interval box = rnd.box(n);
        const auto [lo2, hi2] = Tensor::matmul_interval(M, box.inf(), box.sup());
        aux_case("matmul_interval", [&] {
            check("matmul_interval", Interval(lo2, hi2), lean::Interval(box).linear_map(M));
        });
        aux_case("Interval::linear_map", [&] {
            // exact linear maps
            check("Interval::linear_map", box.linear_map(M), lean::Interval(box).linear_map(M));
        });
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!test::oracle::available()) return 0;
    const BackendScope lean_backend("lean");
    const lean::DTypeScope nearest("nearest");
    test::oracle::Random rnd(11);
    aux_elementwise(rnd);
    aux_sums(rnd);
    aux_rearrangements(rnd);
    aux_interval_products(rnd);
    return test::finish("lean tensor primitives against the oracle");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
