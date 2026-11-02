// test_lean_ops - every lean::Interval and lean::Zonotope method against the CORALean oracle
//
// Calls each method once on the examples of the oracle's README and checks the shape of the answer
// (and the values that are known by hand), so that a change of an op's wire form shows up here.
// Skips without CORACPP_ORACLE.

#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"
#include "oracle_testing.h"
#include "testing.h"

using namespace cora;

namespace {

/// A tensor (rows, cols) of the numbers, row-major.
Tensor aux_matrix(int64_t rows, int64_t cols, const std::vector<double> &v) {
    return Tensor::from_data(v, {rows, cols});
}

/// Whether the box has the given number of rows and columns.
bool aux_shape(const lean::Interval &I, int64_t rows, int64_t cols) {
    return I.inf().rows() == rows && I.inf().cols() == cols;
}

bool aux_shape(const lean::Zonotope &Z, int64_t n, int64_t m) {
    return Z.center().rows() == n && Z.generators().cols() == m;
}

} // namespace

int main() {
    if (!test::oracle::available()) return 0;
    const lean::DTypeScope binary64("binary64");
    const lean::Interval I(aux_matrix(2, 1, {1, -2}), aux_matrix(2, 1, {3, 4}));
    const lean::Interval J(aux_matrix(2, 1, {0.5, -1}), aux_matrix(2, 1, {2, 3}));
    const lean::Interval P(aux_matrix(2, 1, {0.5, 1}), aux_matrix(2, 1, {2, 3}));  // positive
    const lean::Interval A(aux_matrix(2, 2, {1, 2, 3, 4}), aux_matrix(2, 2, {2, 3, 4, 5}));
    const Tensor M = aux_matrix(2, 2, {1, 2, 0, 1}), d = aux_matrix(2, 1, {1, 1});
    const lean::Zonotope Z(aux_matrix(2, 1, {1, 2}), aux_matrix(2, 3, {1, 0.5, 0, 0, 1, 2}));
    const lean::Zonotope Z2(aux_matrix(2, 1, {0, 1}), aux_matrix(2, 2, {2, 0, 0, 1}));
    const lean::Interval U(aux_matrix(2, 1, {-0.5, -0.5}), aux_matrix(2, 1, {0.5, 0.5}));
    const lean::Interval K(aux_matrix(2, 1, {1.5, 0}), aux_matrix(2, 1, {2.5, 1}));  // inside I
    const lean::Interval G(aux_matrix(2, 1, {1.5, 2}), aux_matrix(2, 1, {3, 4}));    // above 1

    const lean::Zonotope Zin(aux_matrix(2, 1, {1, 2}), aux_matrix(2, 2, {1, 0, 0, 1}));

    // Interval: the same box out for the elementwise operations -------------------------------
    for (const lean::Interval &R : {I.abs(), I.uminus(), I.uplus(), I.sign(), I.round(), P.sqrt(),
                                    I.exp(), I.sin(), I.cos(), U.tan(), I.sinh(), I.cosh(),
                                    I.tanh(), P.log(), U.asin(), U.acos(), I.atan(), I.asinh(),
                                    G.acosh(), U.atanh(), I.atan2(P), I.enlarge(2), I.power(3),
                                    I.rdivide(P), I.mrdivide(4), I.plus(J), I.minus(J), I.times(J),
                                    I.unite(J), I.convex_hull(J), I.mink_diff(J), I.min(J),
                                    I.max(J), I.intersect(J), I.enclose(J), I.reduce(1)})
        test::check(aux_shape(R, 2, 1), "an elementwise operation keeps the shape");
    test::check(I.plus(J).inf().gather().data()[0] == 1.5, "plus adds the infima");
    test::check(I.cart_prod(J).inf().rows() == 4, "cart_prod stacks");
    test::check(aux_shape(I.linear_map(M), 2, 1) && aux_shape(I.linear_map(A), 2, 1),
                "linear_map by a matrix and by an interval matrix");
    test::check(aux_shape(I.quad_map({M, M}), 2, 1), "quad_map, one row per matrix");

    // Interval: predicates and numbers -------------------------------------------------------
    test::check(!I.le(J) && !I.lt(J) && !I.isequal(J) && I.isequal(I), "le, lt, isequal");
    test::check(I.is_full_dim() && I.is_bounded() && !I.representsa("emptySet"), "is_full_dim...");
    test::check(I.contains(K).proved && !K.contains(I).proved, "contains is a proof");
    test::check(I.contains(Z).value == I.contains(Z).proved, "contains a zonotope");
    test::check(!I.is_intersecting(J).value == I.is_intersecting(J).proved &&
                    !I.is_intersecting(J).proved, "is_intersecting");
    test::check(I.center().rows() == 2 && I.rad().rows() == 2 && I.infimum().rows() == 2 &&
                    I.supremum().rows() == 2,
                "center, rad, infimum, supremum");
    test::check(I.radius() > 0 && I.norm() > 0 && I.volume() == 12, "radius, norm, volume");
    test::check(I.support_function(d) == 7, "support_function");

    // Interval: selection, reshaping, matrices ------------------------------------------------
    test::check(aux_shape(I.project({1}), 1, 1) && aux_shape(I.lift(4, {3, 1}), 4, 1) &&
                    aux_shape(I.project_high_dim(4, {3, 1}), 4, 1) &&
                    aux_shape(I.partition(4, {1, 3}), 2, 1) && aux_shape(I.dirac(1), 2, 1),
                "project, lift, project_high_dim, partition, dirac");
    const auto [lower, upper] = I.split(0);
    test::check(aux_shape(lower, 2, 1) && aux_shape(upper, 2, 1), "split");
    test::check(I.boundary_point(d, aux_matrix(2, 1, {2, 1})).has_value(), "boundary_point");
    test::check(aux_shape(A.transpose(), 2, 2) && aux_shape(A.ctranspose(), 2, 2) &&
                    aux_shape(A.prod(), 2, 1) && aux_shape(I.diag(), 2, 2) &&
                    aux_shape(A.kron(I), 4, 2) && aux_shape(A.horzcat(A), 2, 4) &&
                    aux_shape(A.vertcat(A), 4, 2) && aux_shape(A.sum(1), 2, 1) &&
                    aux_shape(A.cat(A, 1), 4, 2) && aux_shape(A.mpower(3), 2, 2) &&
                    aux_shape(A.tril(), 2, 2) && aux_shape(A.triu(1), 2, 2),
                "the matrix operations");
    test::check(aux_shape(lean::Interval::empty(2), 2, 1) && aux_shape(lean::Interval::origin(3), 3, 1) &&
                    aux_shape(lean::Interval::unbounded(2), 2, 1) &&
                    aux_shape(lean::Interval::enclose_points(aux_matrix(2, 3, {1, 2, 3, 0, -1, 4})),
                              2, 1) &&
                    aux_shape(lean::Interval::interval_matrix(M, M), 2, 2),
                "empty, origin, unbounded, enclose_points, interval_matrix");
    test::check(aux_shape(I.zonotope(), 2, 2) && I.vertices().V.cols() == 4, "zonotope, vertices");

    // Zonotope: sets ----------------------------------------------------------------------------
    test::check(aux_shape(Z.linear_map(M), 2, 3) && aux_shape(Z.linear_map(A), 2, 5) &&
                    aux_shape(Z.plus(Z2), 2, 5) && aux_shape(Z.plus(I), 2, 3) &&
                    aux_shape(Z.linear_combination(Z2), 2, 6) && aux_shape(Z.exact_plus(Z2, 1), 2, 4) &&
                    aux_shape(Z.reduce(1), 2, 2) && aux_shape(Z.abs(), 2, 3),
                "linear_map, plus, linear_combination, exact_plus, reduce, abs");
    test::check(aux_shape(Z.intersect(Z2), 2, 5) && aux_shape(Z.convex_hull(Z2), 2, 6) &&
                    aux_shape(Z.enclose(Z2), 2, 6) && aux_shape(Z.cart_prod(Z2), 4, 5) &&
                    Z.compact().generators().cols() <= 3,
                "intersect, convex_hull, enclose, cart_prod, compact");
    test::check(aux_shape(Z.intersect_strip(aux_matrix(1, 2, {1, 0}), aux_matrix(1, 1, {0.5}),
                                            aux_matrix(1, 1, {1}), aux_matrix(2, 1, {1, 0})),
                          2, 6),
                "intersect_strip");
        test::check(aux_shape(Z.quad_map({M}), 1, 36) && aux_shape(Zin.cub_map({{M, M}}), 1, 125),
                "quad_map, cub_map");
    test::check(lean::Zonotope::enclose_many({Z, Z2}, d).has_value() &&
                    aux_shape(lean::Zonotope::enclose_points(aux_matrix(2, 3, {1, 2, 3, 0, -1, 4}),
                                                             aux_matrix(2, 2, {1, 0, 0, 1})),
                              2, 2) &&
                    lean::Zonotope::filter_out({Z, Z2}).size() <= 2 &&
                    aux_shape(lean::Zonotope::origin(2), 2, 0),
                "enclose_many, enclose_points, filter_out, origin");
    test::check(aux_shape(Z.project({1}), 1, 3) && aux_shape(Z.lift(2, {1, 0}), 2, 3) &&
                    aux_shape(Z.project_high_dim(3, {2, 0}), 3, 3) &&
                    aux_shape(Z.rotate(20, 0, 1, 0.5), 2, 5),
                "project, lift, project_high_dim, rotate");
    test::check(aux_shape(Z.split(1).first, 2, 3) && aux_shape(Z.split_first_gen().second, 2, 3) &&
                    aux_shape(Z.split_longest_gen().first, 2, 3),
                "split, split_first_gen, split_longest_gen");

    // Zonotope: predicates and numbers -----------------------------------------------------
    test::check(Z.constr_sat(aux_matrix(1, 2, {1, 0}), aux_matrix(1, 1, {100})) && Z.is_bounded() &&
                    Z.is_full_dim() && Z.isequal(Z) && !Z.representsa("point"),
                "constr_sat, is_bounded, is_full_dim, isequal, representsa");
    test::check(Z.contains(Zin).proved && !Z.is_intersecting(Z2).proved,
                "contains, is_intersecting");
    test::check(Z.norm() > 0 && Z.support_function(d) == 7.5 && Z.radius() > 0 &&
                    Z.minnorm().has_value() && Z.generator_length().rows() == 3,
                "norm, support_function, radius, minnorm, generator_length");
    test::check(Z.dominant_directions().has_value() && Z.vertices().V.cols() == 8 &&
                    aux_shape(Z.interval(), 2, 1),
                "dominant_directions, vertices, interval");
    return test::finish("lean ops");
}
