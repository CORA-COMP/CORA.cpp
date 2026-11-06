// test_lean_zonotope_measures - exact_plus, is_bounded, is_full_dim, isequal, norm, enclose_points,
//
// enclose_many and lift of Zonotope against the CORALean oracle
//
// ORACLE: contSet/@zonotope/exactPlus.m
// ORACLE: contSet/@zonotope/isBounded.m
// ORACLE: contSet/@zonotope/isFullDim.m
// ORACLE: contSet/@zonotope/isequal.m
// ORACLE: contSet/@zonotope/norm_.m
// ORACLE: contSet/@zonotope/enclosePoints.m
// ORACLE: contSet/@zonotope/encloseMany.m
// ORACLE: contSet/@zonotope/lift_.m
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "oracle_testing.h"

#include <cmath>

using namespace cora;


namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The result of a zonotope-valued operation against the oracle's: the same bits at nearest; in
/// the sound mode inside the enclosure, by the interval hulls.
template <class A, class B>
void aux_check_set(const A &mine, const B &theirs) {
    if (oracle::mode() == oracle::Mode::exact) return oracle::check(mine, theirs);
    oracle::check(mine.interval(), theirs.enclosure().interval());
}

/// A number the oracle bounds from above (it rounds up): CORA.cpp's, computed in nearest
/// arithmetic, may exceed it by a few ulp only.
void aux_check_bound(double mine, double bound) {
    test::check(mine <= bound * (1 + 1e-12) + 1e-300,
                oracle::op_name() + ": " + std::to_string(mine) + " exceeds the bound " +
                    std::to_string(bound));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 200}};

    oracle::run("zonotope.exactPlus", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m + 2);
            for (const int k : {0, 2, m + 5})
                aux_check_set(Z.exact_plus(Z2, k),
                              lean::Zonotope(Z).exact_plus(lean::Zonotope(Z2), k));
        }
    });

    oracle::run("zonotope.isBounded", [&](oracle::Random &rnd) {
        const Zonotope Z = rnd.zonotope(3, 5);
        oracle::check(Z.is_bounded(), lean::Zonotope(Z).is_bounded());
    });

    oracle::run("zonotope.isFullDim", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {5, 30}, {4, 2}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            // the oracle proves full dimension or says nothing: a proof is never contradicted
            if (lean::Zonotope(Z).is_full_dim())
                test::check(Z.is_full_dim(), "isFullDim: proved, so true");
            if (m < n) test::check(!Z.is_full_dim(), "isFullDim: fewer generators than dimensions");
        }
    });

    oracle::run("zonotope.isequal", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            // the oracle's test is structural and certified when true; CORA's accepts more
            test::check(Z.isequal(Z) && lean::Zonotope(Z).isequal(lean::Zonotope(Z)),
                        "isequal: a set equals itself");
            test::check(!Z.isequal(Z2) && !lean::Zonotope(Z).isequal(lean::Zonotope(Z2)),
                        "isequal: two random sets differ");
        }
    });

    oracle::run("zonotope.norm", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            // the oracle returns the square of the bound; its sum of squares is CORA.cpp's, so
            // the root of it is the same double
            const double bound = std::sqrt(lean::Zonotope(Z).norm());
            if (oracle::mode() == oracle::Mode::exact)
                oracle::check(Z.norm().item(), bound);
            else
                aux_check_bound(Z.norm().item(), bound);
        }
    });

    oracle::run("zonotope.enclosePoints", [&](oracle::Random &rnd) {
        for (const int n : {2, 3}) {
            const Tensor points = rnd.matrix(n, 20);
            // the oracle takes the orthogonal frame; both use the one of the identity here
            const Tensor U = points.eye_like(n);
            aux_check_set(Zonotope::enclose_points(points, U),
                          lean::Zonotope::enclose_points(points, U, "stursberg"));
        }
    });

    oracle::run("zonotope.encloseMany", [&](oracle::Random &rnd) {
        for (const int n : {2, 3}) {
            std::vector<Zonotope> Zs;
            std::vector<lean::Zonotope> theirs;
            for (int k = 0; k < 3; ++k) {
                Zs.push_back(rnd.zonotope(n, n));
                theirs.emplace_back(Zs.back());
            }
            const Tensor direction = rnd.direction(n);
            const auto reference = lean::Zonotope::enclose_many(theirs, direction);
            test::check(reference.has_value(), "encloseMany: the oracle found a box");
            // the oracle reduces to order 1 with its own Girard and takes a boxed inverse
            if (reference) aux_check_set(Zonotope::enclose_many(Zs, direction).first, *reference);
        }
    });

    oracle::run("zonotope.lift", [&](oracle::Random &rnd) {
        const Zonotope Z = rnd.zonotope(4, 6);
        const std::vector<int64_t> dims{1, 2, 3, 0};
        aux_check_set(Z.lift(4, dims), lean::Zonotope(Z).lift(4, dims));
    });
    return test::finish("lean zonotope measures");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
