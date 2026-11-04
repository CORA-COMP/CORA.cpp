// test_lean_ellipsoid - the operations of Ellipsoid that the CORALean oracle serves
//
// Bit for bit: center (the stored q), cartProd with a point (exact: Q padded with zero rows and
// columns, q extended), and the Booleans isFullDim and isBigger on well-conditioned ellipsoids,
// where CORA's rank and eigenvalue tests and CORALean's certified Cholesky checks agree (CORA's
// tolerances 100 eps and 1e-6 do not enter, and CORA's shortcut for dimension 1 is not taken;
// docs/cora-differences.md). The rest of the batch has no oracle op: ellipsoid-ellipsoid cartProd,
// contains, and, enclose, enclosePoints, generators, intersectStrip, ellipsoidNorm, isBadDir,
// interval.
//
// ORACLE: contSet/@ellipsoid/center.m
// ORACLE: contSet/@ellipsoid/cartProd_.m
// ORACLE: contSet/@ellipsoid/isFullDim.m
// ORACLE: contSet/@ellipsoid/isBigger.m

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/oracle/ellipsoid.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// A random positive definite ellipsoid of n dimensions: Q = A A' + I/2 (symmetric to the bit)
/// and a normal center.
Ellipsoid aux_ellipsoid(oracle::Random &rnd, int64_t n, double shrink = 1.0) {
    const Tensor A = rnd.matrix(n, n);
    const Tensor Q = (A.matmul(A.matrix_transpose()) + A.eye_like(n) * 0.5) * shrink;
    return {(Q + Q.matrix_transpose()) * 0.5, rnd.vector(n)};
}

/// A degenerate ellipsoid of rank n - 1: Q = A A' with A of n x (n - 1).
Ellipsoid aux_flat_ellipsoid(oracle::Random &rnd, int64_t n) {
    const Tensor A = rnd.matrix(n, n - 1);
    const Tensor Q = A.matmul(A.matrix_transpose());
    return {(Q + Q.matrix_transpose()) * 0.5, rnd.vector(n)};
}

} // namespace

int main() {
    if (!oracle::available()) return 0;
    oracle::run("ellipsoid.center", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3);
        oracle::check(E.center(), lean::Ellipsoid(E).center());
    });
    oracle::run("ellipsoid.cartProd", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3);
        const Tensor p = rnd.vector(2);
        // the oracle's product with a point is exact: the nominal parts agree to the bit
        const Ellipsoid theirs = lean::Ellipsoid(E).cart_prod(p).nominal();
        const Ellipsoid mine = E.cart_prod(p);
        oracle::check_exact("cartProd.Q", mine.Q(), theirs.Q());
        oracle::check_exact("cartProd.q", mine.q(), theirs.q());
    });
    oracle::run("ellipsoid.isFullDim", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3), flat = aux_flat_ellipsoid(rnd, 3);
        oracle::check(E.is_full_dim(), lean::Ellipsoid(E).is_full_dim());
        oracle::check(flat.is_full_dim(), lean::Ellipsoid(flat).is_full_dim());
    });
    oracle::run("ellipsoid.isBigger", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3);
        // a third of the shape is inside, and E is not inside it
        const Ellipsoid small(E.Q() * (1.0 / 3.0), rnd.vector(3));
        oracle::check(E.is_bigger(small), lean::Ellipsoid(E).is_bigger(lean::Ellipsoid(small)));
        oracle::check(small.is_bigger(E), lean::Ellipsoid(small).is_bigger(lean::Ellipsoid(E)));
        // in dimension 1 CORA.cpp compares (CORA does not)
        const Ellipsoid one(Tensor({{1.0}})), four(Tensor({{4.0}}));
        oracle::check(one.is_bigger(four), lean::Ellipsoid(one).is_bigger(lean::Ellipsoid(four)));
        oracle::check(four.is_bigger(one), lean::Ellipsoid(four).is_bigger(lean::Ellipsoid(one)));
    });
    return test::finish("lean ellipsoid");
}
