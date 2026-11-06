// test_zonotope_unite - zonotope unite: a zonotope that contains the union of zonotopes
//
// CORA: contSet/@zonotope/or.m
// PROPERTY: contSet/@zonotope/or.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Zonotope box(double lo, double hi, double lo2, double hi2) {
    return {column({(lo + hi) / 2, (lo2 + hi2) / 2}),
            Tensor({{(hi - lo) / 2, 0.0}, {0.0, (hi2 - lo2) / 2}})};
}

/// Two boxes [0, 1]^2 and [2, 3] x [0, 1]: every method encloses the box [0, 3] x [0, 1], and
/// linprog and iterative give just that; two points give the segment between them.
void unions_by_hand(const std::string &b) {
    const Zonotope A = box(0, 1, 0, 1), B = box(2, 3, 0, 1);
    for (const char *method : {"linprog", "iterative", "parallelotope"}) {
        const Zonotope U = A.unite(B, method);
        const Interval hull = U.interval();
        check(close(hull.inf(), std::vector<double>{0, 0}, 1e-9) &&
                  close(hull.sup(), std::vector<double>{3, 1}, 1e-9),
              std::string(b) + ": " + method + ": the hull of two boxes is [0, 3] x [0, 1]");
        check(U.contains(A, "exact", 1e-9) && U.contains(B, "exact", 1e-9),
              std::string(b) + ": " + method + ": it contains both");
    }
    const Zonotope p(column({0.0, 0.0}), Tensor::zeros({2, 0})),
        q(column({2.0, 0.0}), Tensor::zeros({2, 0}));
    const Zonotope S = p.unite(q);
    check(close(S.center(), std::vector<double>{1, 0}, 1e-9) &&
              S.generators().shape() == std::vector<int64_t>({2, 1}) &&
              close(S.generators().abs(), std::vector<double>{1, 0}, 1e-9),
          b + ": two points give the segment between them");
    const Zonotope same = A.unite(A);
    check(close(same.interval().sup(), std::vector<double>{1, 1}, 1e-9), b + ": a set with itself");
    // no other set
    check(close(A.unite(std::vector<Zonotope>{}).center(), A.center(), 0), b + ": no other set");
}

/// A box and a point enter as zonotopes; three sets take the unpaired one into the next level.
void other_operands_and_odd_counts(const std::string &b) {
    const Zonotope A = box(0, 1, 0, 1);
    const Interval I(column({2.0, 0.0}), column({3.0, 1.0}));
    check(close(A.unite(I).interval().sup(), std::vector<double>{3, 1}, 1e-9), b + ": a box");
    check(close(A.unite(column({5.0, 0.5})).interval().sup(), std::vector<double>{5, 1}, 1e-9),
          b + ": a point");
    const Zonotope C = box(-4, -3, 0, 1);
    for (const char *method : {"linprog", "iterative", "parallelotope"}) {
        const Zonotope U = A.unite(std::vector<Zonotope>{box(2, 3, 0, 1), C}, method);
        check(U.contains(A, "exact", 1e-9) && U.contains(C, "exact", 1e-9) &&
                  U.contains(box(2, 3, 0, 1), "exact", 1e-9),
              std::string(b) + ": " + method + ": three sets");
    }
    // the registry's unite (CORA's or) is the same operation
    const Set S = unite(A, box(2, 3, 0, 1));
    check(close(S.get().interval().sup(), std::vector<double>{3, 1}, 1e-9), b + ": the registry");
}

/// Stated property: the result contains every set of the union, for random zonotopes, also when
/// they are flat.
void union_contains_the_sets(const std::string &b) {
    cora::Rng rng(131);
    for (const auto &[n, m] : {std::pair{2, 3}, {3, 4}}) {
        std::vector<Zonotope> Zs;
        Zs.reserve(3);
        for (int k = 0; k < 3; ++k)
            Zs.push_back(Zonotope::generate_random(rng, n, std::nullopt, m));
        for (const char *method : {"linprog", "iterative", "parallelotope"}) {
            const Zonotope U = Zs[0].unite(std::vector<Zonotope>{Zs[1], Zs[2]}, method);
            for (const Zonotope &Z : Zs)
                check(U.contains(Z, "exact", 1e-8),
                      std::string(b) + ": " + method + ": it contains a set");
        }
    }
    // flat sets: a segment and a point in the plane
    const Zonotope seg(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const Zonotope pt(column({3.0, 0.0}), Tensor::zeros({2, 0}));
    const Zonotope U = seg.unite(pt);
    check(U.contains(seg, "exact", 1e-8) && U.contains(pt, "exact", 1e-8), b + ": flat sets");
}

/// Unsupported and unknown methods, an order, other dimensions, an unbounded box and a batch are
/// refused; with the checks off the union is the same.
void checks_and_their_switch() {
    const Zonotope A = box(0, 1, 0, 1), B = box(2, 3, 0, 1);
    for (const char *method : {"tedrake", "althoff"})
        check(test::raises([&] { A.unite(B, method); }, Error::not_supported),
              std::string("unite: ") + method);
    check(test::raises([&] { A.unite(B, "convex"); }, Error::wrong_value), "unite: unknown method");
    check(test::raises([&] { A.unite(B, "linprog", 2); }, Error::not_supported), "unite: an order");
    check(test::raises([&] { A.unite(Zonotope::origin(3)); }, Error::dimension_mismatch),
          "unite: dimension");
    const double inf = std::numeric_limits<double>::infinity();
    const Interval unbounded(column({0.0, 0.0}), column({inf, 1.0}));
    check(test::raises([&] { A.unite(unbounded); }, Error::wrong_value), "unite: an unbounded box");
    check(test::raises([&] { Zonotope::stack({A, B}).unite(A); }, Error::not_supported),
          "unite: a batch");
    const Tensor on = A.unite(B, "iterative").generators();
    ChecksScope off(false);
    check(close(A.unite(B, "iterative").generators(), on, 0),
          "unite: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        unions_by_hand(b);
        other_operands_and_odd_counts(b);
        union_contains_the_sets(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
