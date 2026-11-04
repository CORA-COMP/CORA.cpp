// test_lean_cont_set - the operations every set has (ContSet) on a box and a zonotope against the
// CORALean oracle
//
// The oracle serves them per class (`interval.*`, `zonotope.*`); the generic entry of CORA.cpp must
// give the same numbers as the class's own. It has no `zonotope.enlarge`, `zonotope.minus`,
// `zonotope.minkDiff` or `projectOnHyperplane`: the property tests of tests/cont_set/cont_set
// cover those.
//
// ORACLE: contSet/@contSet/center.m
// ORACLE: contSet/@contSet/isBounded.m
// ORACLE: contSet/@contSet/isFullDim.m
// ORACLE: contSet/@contSet/reduce.m
// ORACLE: contSet/@contSet/mtimes.m
// ORACLE: contSet/@contSet/enlarge.m
// ORACLE: contSet/@contSet/minus.m
// ORACLE: contSet/@contSet/minkDiff.m
// ORACLE: contSet/@contSet/eq.m
// ORACLE: contSet/@contSet/ne.m
// ORACLE: contSet/@contSet/decompose.m
// ORACLE: contSet/@contSet/projectHighDim_.m
// ORACLE: contSet/@contSet/and_.m
// ORACLE: contSet/@contSet/contains_.m
// ORACLE: contSet/@contSet/isIntersecting_.m
// ORACLE: contSet/@contSet/supportFunc.m
// ORACLE: contSet/@contSet/supportFunc_.m
// ORACLE: contSet/@contSet/uminus.m
// ORACLE: contSet/@contSet/uplus.m
// ORACLE: contSet/@contSet/times.m

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// A zonotope-valued result against the oracle's: the same bits at nearest; in the sound mode
/// inside the enclosure, by the interval hulls.
template <class A, class B>
void aux_check_set(const A &mine, const B &theirs) {
    if (oracle::mode() == oracle::Mode::exact) return oracle::check(mine, theirs);
    oracle::check(mine.interval(), theirs.enclosure().interval());
}

const ContSet &aux_set(const Interval &I) { return I; }
const ContSet &aux_set(const Zonotope &Z) { return Z; }

} // namespace

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
        oracle::check(aux_set(Z).is_bounded(), lean::Zonotope(Z).is_bounded());
    });

    oracle::run("isFullDim", [&](oracle::Random &rnd) {
        const Interval I = rnd.box(4);
        oracle::check(aux_set(I).is_full_dim(0.0), lean::Interval(I).is_full_dim(0.0));
        if (oracle::mode() == oracle::Mode::sound) {
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
                aux_check_set(aux_set(Z).reduce(order).as<Zonotope>(),
                              lean::Zonotope(Z).reduce(order));
        }
    });

    oracle::run("mtimes", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Tensor M = rnd.matrix(n, n);
            const Interval I = rnd.box(n);
            oracle::check(aux_set(I).linear_map(M).as<Interval>(), lean::Interval(I).linear_map(M));
            const Zonotope Z = rnd.zonotope(n, m);
            aux_check_set(aux_set(Z).linear_map(M).as<Zonotope>(), lean::Zonotope(Z).linear_map(M));
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

    oracle::run("interval.minkDiff", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            // S narrower than I, so the difference is not empty
            const Interval I = rnd.box(n), S = I.enlarge(0.3);
            const Interval R = aux_set(I).minkowski_difference(S).as<Interval>();
            const lean::Interval O = lean::Interval(I).mink_diff(lean::Interval(S));
            if (oracle::mode() == oracle::Mode::exact) {
                oracle::check_exact("interval.minkDiff", R, O);
            } else {
                const auto [inf, sup] = O.gather();
                oracle::check_enclosed("interval.minkDiff", Interval(inf, sup), lean::Interval(R));
            }
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
            aux_check_set(aux_set(Z).project_high_dim(2 * n + 1, proj).as<Zonotope>(),
                          lean::Zonotope(Z).project_high_dim(2 * n + 1, proj));
        }
    });

    oracle::run("project", [&](oracle::Random &rnd) {
        for (const int n : {3, 5}) {
            const std::vector<std::array<int64_t, 2>> blocks{{0, 1}, {2, n - 1}};
            const Interval I = rnd.box(n);
            const Zonotope Z = rnd.zonotope(n, 2 * n);
            const std::vector<Set> parts = aux_set(I).decompose(blocks),
                                   zparts = aux_set(Z).decompose(blocks);
            for (std::size_t k = 0; k < blocks.size(); ++k) {
                std::vector<int64_t> dims;
                for (int64_t d = blocks[k][0]; d <= blocks[k][1]; ++d) dims.push_back(d);
                oracle::check(parts[k].as<Interval>(), lean::Interval(I).project(dims));
                aux_check_set(zparts[k].as<Zonotope>(), lean::Zonotope(Z).project(dims));
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

    oracle::run("interval.isIntersecting", [](oracle::Random &rnd) {
        for (const int n : {2, 5, 20}) {
            const Interval I = rnd.box(n), J = rnd.box(n), K = I.enlarge(0.5);
            for (const Interval *S : {&J, &K, &I})
                oracle::check(cora::detail::is_intersecting_impl(I, *S, 0.0),
                              lean::Interval(I).is_intersecting(lean::Interval(*S), 0.0));
        }
    });

    oracle::run("supportFunc", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
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
            aux_check_set(aux_set(Z).uminus().as<Zonotope>(), lean::Zonotope(Z).linear_map(M));
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
            oracle::check(aux_set(I).times(f).as<Interval>(),
                          lean::Interval(I).times(lean::Interval(Interval(f, f))));
            const Zonotope Z = rnd.zonotope(n, m);
            aux_check_set(aux_set(Z).times(f).as<Zonotope>(),
                          lean::Zonotope(Z).linear_map(f.diag()));
        }
    });

    return test::finish("lean cont_set");
}
