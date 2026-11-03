// test_lean_zonotope_set_operations - abs, box, cart_prod, compact, constraints_satisfied, enclose,
// convex_hull, contains and dominant_directions of Zonotope against the CORALean oracle
//
// ORACLE: contSet/@zonotope/abs.m
// ORACLE: contSet/@zonotope/box.m
// ORACLE: contSet/@zonotope/cartProd_.m
// ORACLE: contSet/@zonotope/compact_.m
// ORACLE: contSet/@zonotope/constrSat.m
// ORACLE: contSet/@zonotope/contains_.m
// ORACLE: contSet/@zonotope/convHull_.m
// ORACLE: contSet/@zonotope/dominantDirections.m
// ORACLE: contSet/@zonotope/enclose.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
namespace oracle = test::oracle;

namespace {

/// The columns of an (n, k) matrix as a sorted list, so that two matrices with the same columns in
/// another order compare equal.
std::vector<std::vector<double>> aux_sorted_columns(const std::vector<double> &data, int64_t n,
                                                    int64_t k) {
    std::vector<std::vector<double>> columns(k, std::vector<double>(n));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j) columns[j][i] = data[i * k + j];
    std::sort(columns.begin(), columns.end());
    return columns;
}

/// The result of a zonotope-valued operation against the oracle's: the same bits at nearest; in the
/// sound mode inside the enclosure, by the interval hulls (the oracle's certified containment of a
/// zonotope in a zonotope is too slow for large ones).
template <class A, class B>
void aux_check_set(const A &mine, const B &theirs) {
    if (oracle::mode() == oracle::Mode::exact) return oracle::check(mine, theirs);
    oracle::check(mine.interval(), theirs.enclosure().interval());
}

/// enclose and convex_hull: CORA lists the generators (G+G2)/2, (c-c2)/2, (G-G2)/2, CORALean's
/// linComb (G+G2)/2, (G-G2)/2, (c-c2)/2: the same center and the same generators, in another order.
void aux_check_same_generators(const Zonotope &mine, const lean::Zonotope &theirs) {
    if (oracle::mode() != oracle::Mode::exact) return aux_check_set(mine, theirs);
    const Zonotope other = theirs.gather().first;
    const int64_t n = mine.dim(), m = mine.generators().shape().back();
    test::check(other.generators().shape().back() == m, oracle::op_name() + ": generator count");
    test::check(oracle::same_bits(oracle::op_name() + ".c", mine.center(), other.center()),
                oracle::op_name() + ": the same center");
    test::check(aux_sorted_columns(mine.generators().data(), n, m) ==
                    aux_sorted_columns(other.generators().data(), n, m),
                oracle::op_name() + ": the same generators, in another order");
}

/// The interval hull of CORA.cpp's cubic map against the oracle's: CORA lists N(N+1)(N+2)/6
/// generators and sums them in its own order, CORALean N^3 with the closed form of the sums, so
/// the bits differ but the box they span agrees to a relative 1e-10 (a few hundred ulp of the sums
/// of cubes).
void aux_check_hull(const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval box = mine.interval();
    const auto [inf, sup] = theirs.interval().gather();
    const std::vector<double> a = box.inf().data(), b = inf.data(), c = box.sup().data(),
                              d = sup.data();
    bool ok = a.size() == b.size();
    for (std::size_t i = 0; ok && i < a.size(); ++i)
        ok = std::abs(a[i] - b[i]) <= 1e-10 * (1 + std::abs(b[i])) &&
             std::abs(c[i] - d[i]) <= 1e-10 * (1 + std::abs(d[i]));
    test::check(ok, oracle::op_name() + ": the box of the oracle's cubic map");
}

} // namespace

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 200}};
    oracle::run("zonotope.abs", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            aux_check_set(Z.abs(), lean::Zonotope(Z).abs());
        }
    });
    oracle::run("zonotope.box", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            oracle::check(Z.box().interval(), lean::Zonotope(Z).interval());
        }
    });
    oracle::run("zonotope.cartProd", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n + 1, m + 2);
            aux_check_set(Z.cart_prod(Z2), lean::Zonotope(Z).cart_prod(lean::Zonotope(Z2)));
        }
    });
    oracle::run("zonotope.compact", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor &G = Z.generators();
            // a generator of zeros in the middle and at the end
            const Tensor zero = G.slice(-1, 0, 1).zeros_like();
            const Zonotope Zz(
                Z.center(), Tensor::concat({G.slice(-1, 0, 2), zero, G.slice(-1, 2, m), zero}, -1));
            aux_check_set(Zz.compact("zeros", 0.0), lean::Zonotope(Zz).compact("zeros", 0.0));
        }
    });
    oracle::run("zonotope.constrSat", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor C = rnd.matrix(3, n);
            const Tensor sup = Z.linear_map(C).interval().sup();
            for (const double margin : {1e-3, -1e-3}) {
                const Tensor d = sup + margin;
                oracle::check(Z.constraints_satisfied(C, d), lean::Zonotope(Z).constr_sat(C, d));
            }
        }
    });
    oracle::run("zonotope.enclose", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            aux_check_same_generators(Z.enclose(Z2), lean::Zonotope(Z).enclose(lean::Zonotope(Z2)));
        }
    });
    oracle::run("zonotope.convexHull", [&](oracle::Random &rnd) {
        for (const auto [n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            aux_check_same_generators(Z.convex_hull(Z2),
                                      lean::Zonotope(Z).convex_hull(lean::Zonotope(Z2)));
        }
    });
    oracle::run("zonotope.contains", [&](oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{2, 3}, {3, 5}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            // half of Z about its center is in Z, one and a half times is not
            const Zonotope inner(Z.center(), Z.generators() * 0.5);
            const Zonotope outer(Z.center(), Z.generators() * 1.5);
            oracle::check(Z.contains(inner, "approx:st"),
                          lean::Zonotope(Z).contains(lean::Zonotope(inner)));
            oracle::check(Z.contains(outer, "approx:st"),
                          lean::Zonotope(Z).contains(lean::Zonotope(outer)));
        }
    });
    oracle::run("zonotope.cubMap", [&](oracle::Random &rnd) {
        for (const auto [n, m] : {std::pair{2, 2}, {3, 3}, {3, 6}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            std::vector<std::vector<Tensor>> T(2);
            std::vector<std::vector<lean::Tensor>> theirs_T(2);
            for (std::size_t i = 0; i < 2; ++i)
                for (int k = 0; k < n; ++k) {
                    T[i].push_back(rnd.matrix(n, n));
                    theirs_T[i].emplace_back(T[i].back());
                }
            aux_check_hull(Z.cubic_map(T), lean::Zonotope(Z).cub_map(theirs_T));
        }
    });
    oracle::run("zonotope.dominantDirections", [&](oracle::Random &rnd) {
        if (oracle::mode() != oracle::Mode::exact) return; // a choice of columns has no enclosure
        for (const auto [n, m] : {std::pair{2, 3}, {3, 9}, {4, 12}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const auto theirs = lean::Zonotope(Z).dominant_directions();
            test::check(theirs.has_value(), "dominantDirections: the oracle found directions");
            test::check(aux_sorted_columns(Z.dominant_directions().data(), n, n) ==
                            aux_sorted_columns(theirs->gather().data(), n, n),
                        "dominantDirections: the same generators");
        }
    });
    return test::finish("lean zonotope set operations");
}
