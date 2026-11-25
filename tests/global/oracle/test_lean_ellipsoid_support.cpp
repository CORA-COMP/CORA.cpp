// test_lean_ellipsoid_support - support values, point differences and point clouds of Ellipsoid
//
// against the CORALean oracle, bit for bit
//
// support_function is d'q + sqrt(d'(Q d)) in CORALean's order (Ellipsoid.supportFuncUp; at nearest
// the root is the machine's); mink_diff with a point is q - p (Ellipsoid.minkDiffPoint);
// enclose_points ("cov", a cloud that spans the space) is G G' with G = T U diag(nt sqrt(maxDist))
// at CORA.cpp's own rotations T and U (Ellipsoid.enclosePointsCov, the rotations its certificate);
// dim and isnan read the set. The difference of two ellipsoids and the degenerate clouds have no
// oracle op.
//
// ORACLE: contSet/@ellipsoid/supportFunc_.m
// ORACLE: contSet/@ellipsoid/minkDiff.m
// ORACLE: contSet/@ellipsoid/enclosePoints.m
// ORACLE: contSet/@ellipsoid/dim.m
// ORACLE: contSet/@ellipsoid/isnan.m
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/oracle/ellipsoid.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A random positive definite ellipsoid of n dimensions: Q = A A' + I/2 (symmetric to the bit).
Ellipsoid aux_ellipsoid(oracle::Random &rnd, int64_t n) {
    const Tensor A = rnd.matrix(n, n);
    const Tensor Q = A.matmul(A.matrix_transpose()) + A.eye_like(n) * 0.5;
    return {rnd.vector(n).squeeze(-1), (Q + Q.matrix_transpose()) * 0.5};
}

/// The rotations T and U of Ellipsoid::enclose_points for the columns P (n, p), computed by the
/// same operations: T of the centered cloud, U of the covariance of the rotated cloud.
std::pair<Tensor, Tensor> aux_rotations(const Tensor &P) {
    const Tensor centered = P - P.mean(1, true);
    const Tensor T = centered.svd()[0];
    const Tensor rotated = T.matrix_transpose().matmul(centered);
    const Tensor deviations = rotated - rotated.mean(1, true);
    const Tensor covariance =
        deviations.matmul(deviations.matrix_transpose()) / double(P.shape()[1] - 1);
    return {T, covariance.svd()[0]};
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    // the upper support value along random directions
    oracle::run("ellipsoid.supportFunc", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3, 5}) {
            const Ellipsoid E = aux_ellipsoid(rnd, n);
            const Tensor d = rnd.direction(n);
            oracle::check(E.support_function(d), lean::Ellipsoid(E).support_function(d));
        }
    });
    // the translate by a point
    oracle::run("ellipsoid.minkDiff", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 4}) {
            const Ellipsoid E = aux_ellipsoid(rnd, n);
            const Tensor p = rnd.point(n);
            const Ellipsoid theirs = lean::Ellipsoid(E).mink_diff(p).nominal();
            const Ellipsoid mine = E.mink_diff(p);
            oracle::check("minkDiff.Q", mine.Q(), theirs.Q());
            oracle::check("minkDiff.q", mine.q(), theirs.q());
        }
    });
    // the ellipsoid of a cloud that spans the space, at CORA.cpp's rotations
    oracle::run("ellipsoid.enclosePoints", [](oracle::Random &rnd) {
        for (const int64_t n : {2, 3}) {
            const Tensor P = rnd.matrix(n, 4 * n);
            const auto [T, U] = aux_rotations(P);
            const Ellipsoid mine = Ellipsoid::enclose_points(P.matrix_transpose());
            const Ellipsoid theirs = lean::Ellipsoid::enclose_points(P, T, U).nominal();
            oracle::check("enclosePoints.Q", mine.Q(), theirs.Q());
            oracle::check("enclosePoints.q", mine.q(), theirs.q());
        }
    });
    // the dimension and the NaN test
    oracle::run("ellipsoid.dim", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3);
        oracle::check(E.dim(), lean::Ellipsoid(E).dim());
    });
    oracle::run("ellipsoid.isnan", [](oracle::Random &rnd) {
        const Ellipsoid E = aux_ellipsoid(rnd, 3);
        oracle::check(E.isnan(), lean::Ellipsoid(E).isnan());
    });
    return test::finish("lean ellipsoid support");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
