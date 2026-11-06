// test_polytope_compact - Polytope::compact: the minimal representation, by method
//
// CORA: contSet/@polytope/compact_.m
// CORA: contSet/@polytope/private/priv_compact_zeros.m
// CORA: contSet/@polytope/private/priv_compact_toEquality.m
// CORA: contSet/@polytope/private/priv_compact_alignedIneq.m
// CORA: contSet/@polytope/private/priv_compact_alignedEq.m
// CORA: contSet/@polytope/private/priv_compact_1D.m
// CORA: contSet/@polytope/private/priv_compact_2D.m
// CORA: contSet/@polytope/private/priv_compact_nD.m
// CORA: contSet/@polytope/private/priv_compact_all.m
// PROPERTY: contSet/@polytope/compact_.m
//
// Property: the compacted set is the same set (same points inside) with fewer rows
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The number of rows of A and Ae.
int64_t rows(const Polytope &P) { return P.A().shape()[0] + P.Ae().shape()[0]; }

/// Redundant rows: aligned, zero, and implied by the others.
Polytope redundant() {
    return Polytope(
        Tensor({{1.0, 0.0}, {2.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {1.0, 1.0}}),
        column({1.0, 4.0, 1.0, 0.0, 0.0, 5.0}));
}

/// Every method of the halfspaces.
void methods(const std::string &b) {
    const Polytope P = redundant();
    const Tensor inside = Tensor({{0.5, 0.0, 1.0, 0.2}, {0.5, 1.0, 0.0, 0.9}});
    for (const char *method : {"all", "A", "aligned", "zeros", "AtoAe"}) {
        const Polytope C = P.compact(method);
        check(test::inside_halfspaces(inside, C.constraints()),
              b + ": " + method + ": same points");
        check(!C.contains(column({1.5, 0.5})) && !C.contains(column({0.5, -0.5})),
              b + ": " + method + ": same outside");
    }
    check(rows(P.compact("all")) == 4, b + ": all: the 4 facets of the square");
    check(rows(P.compact("A")) == 4, b + ": A: the 4 facets");
    check(rows(P.compact("aligned")) == 5, b + ": aligned: the doubled row goes");
    // a zero row 0 <= b holds everywhere and goes
    const Polytope zero(Tensor({{0.0, 0.0}, {1.0, 0.0}}), column({1.0, 1.0}));
    check(rows(zero.compact("zeros")) == 1, b + ": zeros: the row 0 <= 1 goes");
    check(zero.compact("zeros").compact("zeros").A().shape()[0] == 1, b + ": zeros: idempotent");
    check(Polytope(Tensor({{0.0, 0.0}}), column({-1.0}))
                  .compact("zeros")
                  .contains(column({0.0, 0.0})) == false,
          b + ": zeros: 0 <= -1 is the empty set");
    // two opposite rows pin a coordinate: an equality
    const Polytope pinned(Tensor({{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}}),
                          column({1.0, -1.0, 2.0, 0.0}));
    const Polytope pin = pinned.compact("AtoAe");
    check(pin.Ae().shape()[0] == 1 && pin.A().shape()[0] == 2, b + ": AtoAe: x = 1 is an equality");
    check(close(pin.be(), {1}, 1e-9), b + ": AtoAe: its offset");
}

/// Equalities: aligned ones merge, contradicting ones are the empty set.
void equalities(const std::string &b) {
    const Polytope twice(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}),
                         Tensor({{1.0, 1.0}, {2.0, 2.0}}), column({1.0, 2.0}));
    check(rows(twice.compact("Ae")) == 1, b + ": Ae: parallel equalities merge");
    const Polytope clash(Tensor::zeros({0, 2}), Tensor::zeros({0, 1}),
                         Tensor({{1.0, 1.0}, {2.0, 2.0}}), column({1.0, 3.0}));
    check(clash.compact("Ae").V().shape()[1] == 0, b + ": Ae: contradicting equalities: empty");
    check(clash.compact("all").V().shape()[1] == 0, b + ": all: contradicting equalities: empty");
    check(rows(twice.compact("all")) == 1, b + ": all: one equality");
}

/// The dimensions and the special sets: 1D, 2D, nD, R^n, empty, vertices.
void dimensions(const std::string &b) {
    const Polytope line(Tensor({{1.0}, {-1.0}, {2.0}}), column({3.0, 0.0, 5.0}));
    const Polytope line_c = line.compact("all");
    check(rows(line_c) == 2 && line_c.contains(column({1.0})) && !line_c.contains(column({4.0})),
          b + ": 1D: the tight bounds");
    check(Polytope(Tensor({{1.0}, {-1.0}}), column({0.0, -1.0})).compact("all").V().shape()[1] == 0,
          b + ": 1D: x <= 0 and x >= 1 is empty");
    const Polytope cube =
        Polytope(Tensor::concat({Tensor::eye(3), -Tensor::eye(3), Tensor::ones({1, 3})}, 0),
                 column({1, 1, 1, 0, 0, 0, 9}));
    check(rows(cube.compact("all")) == 6, b + ": nD: the redundant row goes");
    check(Polytope(Tensor::concat({Tensor::eye(3), -Tensor::eye(3)}, 0), column({1, 1, 1, 0, 0, 0}))
                  .compact("all")
                  .A()
                  .shape()[0] == 6,
          b + ": nD: a minimal set stays");
    const Polytope whole = Polytope(Tensor({{0.0, 0.0}}), column({1.0})).compact("all");
    check(whole.A().shape()[0] == 0, b + ": no row left: R^n");
    const Polytope sq(Tensor({{0.0, 1.0, 1.0, 0.0}, {0.0, 0.0, 1.0, 1.0}}));
    const Polytope compact_v = sq.compact("V");
    // the extreme vertices
    check(compact_v.is_v_rep() && compact_v.V().shape() == Shape{2, 4},
          b + ": V: the extreme vertices");
    const Polytope with_inner(Tensor({{0.0, 1.0, 0.0, 0.25}, {0.0, 0.0, 1.0, 0.25}}));
    check(with_inner.compact("V").V().shape() == Shape{2, 3}, b + ": V: an inner vertex goes");
    check(Polytope(Tensor({{0.0, 1.0, 0.5}})).compact("V").V().shape() == Shape{1, 2},
          b + ": V in 1D: the extremes");
    check(rows(sq.compact("all")) == 4, b + ": a vertex polytope: its facets");
    check(Polytope::empty(2).compact("all").V().shape()[1] == 0, b + ": the empty set");
    check(test::raises([&] { sq.compact("V").A(); }, Error::special_error), b + ": V stays V");
    // generic dispatch of the set
    const Polytope R = redundant();
    const ContSet &S = R;
    check(S.compact().is<Polytope>() && S.compact("zeros").is<Polytope>(),
          b + ": compacting a set");
}

/// A method that does not exist; V on halfspaces; a tolerance; checks off.
void refusals() {
    const Polytope P = redundant();
    check(test::raises([&] { P.compact("nope"); }, Error::wrong_value),
          "compact: an unknown method");
    check(test::raises([&] { P.compact("V"); }, Error::not_supported), "compact: V needs vertices");
    check(rows(P.compact("all", 1e-6)) == 4, "compact: a given tolerance");
    ChecksScope off(false);
    check(rows(P.compact("all")) == 4, "compact: the same with the checks off");
}

/// Redundant rows (scaled copies, loose rows) change no point: the compacted set holds the same
/// probes, with fewer rows.
void properties(const std::string &b) {
    Rng rng(4);
    for (const int n : {2, 3}) {
        const Polytope Q = test::random_polytope(rng, n, 2 * n);
        const Tensor A =
            Tensor::concat({Q.A(), Q.A().take({0, 1}, 0) * 2.0, Q.A().take({0}, 0)}, 0);
        const Tensor offsets =
            Tensor::concat({Q.b(), Q.b().take({0, 1}, 0) * 2.0, Q.b().take({0}, 0) + 5.0}, 0);
        const Polytope redundant_rows(A, offsets);
        for (const char *method : {"all", "A", "aligned"}) {
            const Polytope C = redundant_rows.compact(method);
            check(C.A().shape()[0] < A.shape()[0], b + ": " + method + ": fewer rows");
            const Tensor probes = test::sample_points(Q.box(), rng, 40) * 1.2;
            bool same = true;
            for (int k = 0; k < 40; ++k) {
                const Tensor p = probes.take({k}, -1);
                same = same && C.contains(p) == Q.contains(p);
            }
            // the same probes for every method
            check(same, b + ": " + method + ": the same probes, " + std::to_string(n) + "D");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        properties(b);
        methods(b);
        equalities(b);
        dimensions(b);
    });
    refusals();
    return test::finish("polytope compact");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
