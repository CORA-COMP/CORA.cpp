// test_lean_cont_set - the operations every set has (ContSet) on a box and a zonotope against the
//
// CORALean oracle
//
// The oracle serves them per class (`interval.*`, `zonotope.*`); the generic entry of CORA.cpp must
// give the same numbers as the class's own. The oracle has no `zonotope.minkDiff`: the property
// tests of tests/cont_set/cont_set cover it. `zonotope.enlarge`, `zonotope.minus` and the two
// `projectOnHyperplane` come with the oracles of batch 5; an older oracle skips them.
//
// ORACLE: contSet/@contSet/center.m
// ORACLE: contSet/@zonotope/center.m
// ORACLE: contSet/@interval/contains_.m
// ORACLE: contSet/@contSet/isBounded.m
// ORACLE: contSet/@contSet/isFullDim.m
// ORACLE: contSet/@contSet/reduce.m
// ORACLE: contSet/@contSet/mtimes.m
// ORACLE: contSet/@contSet/enlarge.m
// ORACLE: contSet/@contSet/minus.m
// ORACLE: contSet/@contSet/minkDiff.m
// ORACLE: contSet/@contSet/projectOnHyperplane.m
// ORACLE: contSet/@contSet/eq.m
// ORACLE: contSet/@contSet/ne.m
// ORACLE: contSet/@contSet/decompose.m
// ORACLE: contSet/@contSet/projectHighDim_.m
// ORACLE: contSet/@contSet/and_.m
// ORACLE: contSet/@contSet/contains_.m
// ORACLE: contSet/@contSet/isIntersecting.m
// ORACLE: contSet/@contSet/isIntersecting_.m
// ORACLE: contSet/@contSet/interval.m
// ORACLE: contSet/@contSet/supportFunc.m
// ORACLE: contSet/@contSet/supportFunc_.m
// ORACLE: contSet/@contSet/uminus.m
// ORACLE: contSet/@contSet/uplus.m
// ORACLE: contSet/@contSet/times.m
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   11-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/capsule.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A projection onto a hyperplane: at nearest the center and the first m generators are
/// CORA.cpp's, bit for bit, and the others are zero.
void aux_check_projected(const Zonotope &mine, const lean::Zonotope &theirs) {
    const Zonotope other = theirs.gather().first;
    const int64_t m = mine.generators().shape().back(), total = other.generators().shape().back();
    test::check(total >= m, "projectOnHyperplane: the generators of the set stay");
    if (total < m) return;
    oracle::check(mine.center(), other.center());
    oracle::check(mine.generators(), other.generators().slice(-1, 0, m));
    test::check(total == m || other.generators().slice(-1, m, total).abs().max().item() == 0,
                "projectOnHyperplane: the other generators are zero at nearest");
}

/// A projection of a box: the same bits at nearest.
void aux_check_projected(const Interval &mine, const lean::Interval &theirs) {
    oracle::check(mine, theirs);
}

const ContSet &aux_set(const Interval &I) { return I; }
const ContSet &aux_set(const Zonotope &Z) { return Z; }

/// Sets of 4 dimensions (from `make`) with unit normals: the axes, and (1/2, 1/2, 1/2, 1/2) in
/// which the sum of squares is exact.
template <class S, class F>
std::vector<std::pair<S, Tensor>> aux_normals(F make) {
    std::vector<std::pair<S, Tensor>> pairs;
    for (const std::vector<double> &c : {std::vector<double>{1, 0, 0, 0},
                                         {0, 0, -1, 0},
                                         {0.5, 0.5, 0.5, 0.5},
                                         {-0.5, 0.5, -0.5, 0.5}})
        pairs.emplace_back(make(4), Tensor::from_data(c, {4, 1}));
    return pairs;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 200}};

    oracle::run("interval.center", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).center(), lean::Interval(I).center());
        }
    });
    oracle::run("zonotope.center", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(aux_set(Z).center(), lean::Zonotope(Z).center_value());
        }
    });

    oracle::run("isBounded", [&](oracle::Random &rnd) {
        const Interval I = rnd.box(3);
        oracle::check(aux_set(I).is_bounded(), lean::Interval(I).is_bounded());
        const Zonotope Z = rnd.zonotope(3, 5);
        // boundedness of each class
        oracle::check(aux_set(Z).is_bounded(), lean::Zonotope(Z).is_bounded());
    });

    oracle::run("isFullDim", [&](oracle::Random &rnd) {
        const Interval I = rnd.box(4);
        oracle::check(aux_set(I).is_full_dim(0.0), lean::Interval(I).is_full_dim(0.0));
        if (oracle::nearest_is_strict()) {
            // a flat box: the first coordinate has no width
            const Tensor &lo = I.inf();
            const Tensor hi = Tensor::concat({lo.slice(-2, 0, 1), I.sup().slice(-2, 1, 4)}, -2);
            const Interval flat(lo, hi);
            oracle::check(aux_set(flat).is_full_dim(0.0), lean::Interval(flat).is_full_dim(0.0));
        }
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {4, 2}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            // the oracle proves full dimension or says nothing: a proof is never contradicted
            if (lean::Zonotope(Z).is_full_dim())
                test::check(aux_set(Z).is_full_dim(), "isFullDim: proved, so true");
            if (m < n) test::check(!aux_set(Z).is_full_dim(), "isFullDim: fewer generators than n");
        }
    });

    oracle::run("reduce", [&](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const int order : {1, 3})
                oracle::check(aux_set(I).reduce(order).as<Interval>(),
                              lean::Interval(I).reduce(order));
        }
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            for (const int order : {1, 2})
                oracle::check(aux_set(Z).reduce(order).as<Zonotope>(),
                              lean::Zonotope(Z).reduce(order));
        }
    });

    oracle::run("mtimes", [&](oracle::Random &rnd) {
        // the linear map of boxes
        for (const auto &[n, m] : sizes) {
            const Tensor M = rnd.matrix(n, n);
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).linear_map(M).as<Interval>(), lean::Interval(I).linear_map(M));
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(aux_set(Z).linear_map(M).as<Zonotope>(), lean::Zonotope(Z).linear_map(M));
        }
    });

    oracle::run("interval.enlarge", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            for (const double factor : {0.0, 0.5, 1.0, 2.5})
                oracle::check(aux_set(I).enlarge(factor).as<Interval>(),
                              lean::Interval(I).enlarge(factor));
        }
    });

    oracle::run("interval.minus", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // a vector is the point box, and I - [p, p] is I + (-p) in every bound
            const Interval I = rnd.box(n);
            const Tensor p = rnd.vector(n);
            oracle::check(aux_set(I).minus(p).as<Interval>(),
                          lean::Interval(I).minus(lean::Interval(Interval(p, p))));
        }
    });

    oracle::run("zonotope.enlarge", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 50}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            for (const double factor : {0.0, 0.5, 1.0, 2.5})
                oracle::check(aux_set(Z).enlarge(factor).as<Zonotope>(),
                              lean::Zonotope(Z).enlarge(factor));
            // a factor per coordinate
            const Tensor column = rnd.vector(n).abs();
            oracle::check(aux_set(Z).enlarge(column).as<Zonotope>(),
                          lean::Zonotope(Z).enlarge(column));
        }
    });

    oracle::run("zonotope.minus", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {20, 50}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor p = rnd.vector(n);
            oracle::check(aux_set(Z).minus(p).as<Zonotope>(), lean::Zonotope(Z).minus(p));
        }
    });

    // the normals are of length 1 exactly: an axis, and (1/2, 1/2, 1/2, 1/2)
    oracle::run("interval.projectOnHyperplane", [](oracle::Random &rnd) {
        for (const auto &[I, c] : aux_normals<Interval>([&](int n) { return rnd.box(n); })) {
            const Tensor d = rnd.vector(1);
            aux_check_projected(aux_set(I).project_on_hyperplane(c, d).as<Interval>(),
                                lean::Interval(I).project_on_hyperplane(c, d.item()));
        }
    });
    oracle::run("zonotope.projectOnHyperplane", [](oracle::Random &rnd) {
        for (const auto &[Z, c] :
             aux_normals<Zonotope>([&](int n) { return rnd.zonotope(n, 8); })) {
            const Tensor d = rnd.vector(1);
            aux_check_projected(aux_set(Z).project_on_hyperplane(c, d).as<Zonotope>(),
                                lean::Zonotope(Z).project_on_hyperplane(c, d.item()));
        }
    });

    oracle::run("interval.minkDiff", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // S narrower than I, so the difference is not empty
            const Interval I = rnd.box(n), S = I.enlarge(0.3);
            const Interval R = aux_set(I).minkowski_difference(S).as<Interval>();
            const lean::Interval O = lean::Interval(I).mink_diff(lean::Interval(S));
            oracle::check("interval.minkDiff", R, O);
        }
    });

    oracle::run("isequal", [&](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n);
            for (const Interval *K : {&I, &J}) {
                oracle::check(eq(I, *K, 0.0), lean::Interval(I).isequal(lean::Interval(*K), 0.0));
                oracle::check(ne(I, *K, 0.0), !lean::Interval(I).isequal(lean::Interval(*K), 0.0));
            }
        }
        for (const auto &[n, m] : sizes) {
            // isequal is structural
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            // the oracle's test is structural and certified when true; CORA's accepts more
            test::check(eq(Z, Z) && lean::Zonotope(Z).isequal(lean::Zonotope(Z)),
                        "isequal: a set equals itself");
            test::check(ne(Z, Z2) && !lean::Zonotope(Z).isequal(lean::Zonotope(Z2)),
                        "isequal: two random sets differ");
        }
    });

    oracle::run("projectHighDim", [&](oracle::Random &rnd) {
        for (const int n : {2, 5}) {
            std::vector<int64_t> proj(n);
            for (int j = 0; j < n; ++j) proj[j] = 2 * (n - 1 - j) + 1;
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).project_high_dim(2 * n + 1, proj).as<Interval>(),
                          lean::Interval(I).project_high_dim(2 * n + 1, proj));
            const Zonotope Z = rnd.zonotope(n, 2 * n);
            oracle::check(aux_set(Z).project_high_dim(2 * n + 1, proj).as<Zonotope>(),
                          lean::Zonotope(Z).project_high_dim(2 * n + 1, proj));
        }
    });

    oracle::run("project", [&](oracle::Random &rnd) {
        for (const int n : {3, 5}) {
            // decompose into blocks
            const std::vector<std::array<int64_t, 2>> blocks{{0, 1}, {2, n - 1}};
            const Interval I = rnd.box(n);
            const Zonotope Z = rnd.zonotope(n, 2 * n);
            const std::vector<Set> parts = aux_set(I).decompose(blocks),
                                   zparts = aux_set(Z).decompose(blocks);
            for (std::size_t k = 0; k < blocks.size(); ++k) {
                std::vector<int64_t> dims;
                for (int64_t d = blocks[k][0]; d <= blocks[k][1]; ++d) dims.push_back(d);
                oracle::check(parts[k].as<Interval>(), lean::Interval(I).project(dims));
                oracle::check(zparts[k].as<Zonotope>(), lean::Zonotope(Z).project(dims));
            }
        }
    });

    oracle::run("interval.and", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // J is I shifted by a third of its width, so the boxes overlap
            const Interval I = rnd.box(n);
            const Tensor shift = (I.sup() - I.inf()) * 0.3;
            const Interval J(I.inf() + shift, I.sup() + shift);
            oracle::check(cora::detail::intersect_impl(I, J).as<Interval>(),
                          lean::Interval(I).intersect(lean::Interval(J)));
        }
    });

    oracle::run("zonotope.contains", [](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 5}}) {
            // a scaled copy about the center lies inside (0.1) or outside (3) a random zonotope
            const Zonotope Z = rnd.zonotope(n, m);
            const Zonotope inner(Z.center(), Z.generators() * 0.1),
                outer(Z.center(), Z.generators() * 3.0);
            for (const Zonotope *S : {&inner, &outer})
                oracle::check(cora::detail::contains_impl(Z, *S, 0.0),
                              lean::Zonotope(Z).contains(lean::Zonotope(*S), 0.0));
        }
    });

    // CORA.cpp's box contains points; the oracle's containment of a box is asked for the point box
    oracle::run("interval.contains", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            const Tensor inside = I.center(), outside = rnd.vector(n) * 3.0;
            const Tensor &corner = I.inf();
            for (const Tensor &p : {inside, corner, outside})
                oracle::check(I.contains(p, 0.0),
                              lean::Interval(I).contains(lean::Interval(Interval(p, p)), 0.0));
        }
    });

    oracle::run("interval.isIntersecting", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), K = I.enlarge(0.5);
            for (const Interval *S : {&J, &K, &I})
                oracle::check(cora::detail::is_intersecting_impl(I, *S, 0.0),
                              lean::Interval(I).is_intersecting(lean::Interval(*S), 0.0));
        }
    });

    // the generic entry: the interval hull of a box, a zonotope and a capsule
    oracle::run("interval", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).interval(), lean::Interval(I));
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(aux_set(Z).interval(), lean::Zonotope(Z).interval());
        }
        const Capsule C = rnd.capsule(3);
        const ContSet &S = C;
        oracle::check(S.interval(), lean::Capsule(C).interval());
    });

    // the generic entry, with its checks: two boxes
    oracle::run("isIntersecting", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), K = I.enlarge(0.5);
            for (const Interval *S : {&J, &K, &I})
                oracle::check(is_intersecting(I, *S, 0.0),
                              lean::Interval(I).is_intersecting(lean::Interval(*S), 0.0));
        }
    });

    oracle::run("supportFunc", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            // support functions of boxes and zonotopes
            const Interval I = rnd.box(n);
            const Tensor d = rnd.direction(n);
            oracle::check(aux_set(I).support_function(d), lean::Interval(I).support_function(d));
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(aux_set(Z).support_function(d), lean::Zonotope(Z).support_function(d));
        }
    });

    // -S is the matrix -I times S (a box: its own negation), S .* f the matrix diag(f) times S
    oracle::run("uminus", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).uminus().as<Interval>(), lean::Interval(I).uminus());
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor M = rnd.vector(n).eye_like(n) * -1.0;
            oracle::check(aux_set(Z).uminus().as<Zonotope>(), lean::Zonotope(Z).linear_map(M));
        }
    });

    oracle::run("uplus", [&](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).uplus().as<Interval>(), lean::Interval(I).uplus());
        }
    });

    oracle::run("times", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Tensor f = rnd.vector(n);
            const Interval I = rnd.box(n);
            // times with a factor
            oracle::check(aux_set(I).times(f).as<Interval>(),
                          lean::Interval(I).times(lean::Interval(Interval(f, f))));
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(aux_set(Z).times(f).as<Zonotope>(),
                          lean::Zonotope(Z).linear_map(f.diag()));
        }
    });

    return test::finish("lean cont_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
