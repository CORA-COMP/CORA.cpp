// test_lean_zonotope_measures_bound - minnorm, intersect_strip and
//
// is_intersecting of Zonotope against the CORALean oracle, within a stated bound
//
// CORA: contSet/@zonotope/minnorm.m
// CORA: contSet/@zonotope/intersectStrip.m
// CORA: contSet/@zonotope/isIntersecting_.m
//
// These cannot match the oracle bit for bit, by construction, so there is no ORACLE tag:
// minnorm takes a square root, which the oracle computes by Newton steps rounded up (one ulp from
// the correctly rounded one); intersect_strip is compared by its box with the oracle's
// `zonotope.intersectStrip`, (I - L C) Z + L strip with an interval matrix, where CORA computes
// c + L (y - C c) (bit for bit against `intersectStripCora` in
// test_lean_zonotope_intersect_strip); is_intersecting is certified one way only (disjointness).
//
// CORA authors: Matthias Althoff, Victor Gassmann, Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   21-November-2026
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

/// The slack of the comparisons: a relative 1e-12 (a few thousand ulp), for the Newton square
/// roots and the rounding of sums of up to 200 products.
constexpr double kSlack = 1e-12;

/// Mine does not exceed the oracle's upper bound, and is within the slack of it.
void aux_below(const std::string &what, double mine, double bound) {
    test::check(mine <= bound * (1 + kSlack),
                what + ": " + std::to_string(mine) + " exceeds the bound " + std::to_string(bound));
    test::check(mine >= bound * (1 - 1e-6), what + ": the bound " + std::to_string(bound) +
                                                " is far above " + std::to_string(mine));
}

/// The interval hull of mine lies in the oracle's enclosure up to the slack, and is not tighter
/// than a relative 1e-9 of the radius from it.
void aux_hull_agrees(const Zonotope &mine, const lean::Zonotope &theirs) {
    const Interval a = mine.interval();
    const auto [inf, sup] = theirs.enclosure().interval().gather();
    const std::vector<double> lo = a.inf().data(), hi = a.sup().data();
    const std::vector<double> blo = inf.data(), bhi = sup.data();
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double scale = 1 + std::abs(bhi[i]) + std::abs(blo[i]);
        test::check(blo[i] <= lo[i] + kSlack * scale && hi[i] <= bhi[i] + kSlack * scale,
                    "intersect_strip: outside the enclosure");
        test::check(lo[i] - blo[i] <= 1e-9 * scale && bhi[i] - hi[i] <= 1e-9 * scale,
                    "intersect_strip: far inside the enclosure");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const std::vector<std::pair<int, int>> sizes{{2, 3}, {5, 30}, {20, 200}};

    oracle::run("zonotope.minnorm", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 4}, {3, 6}, {4, 7}}) {
            const Zonotope Z = rnd.zonotope(n, m);
            const std::optional<double> theirs = lean::Zonotope(Z).minnorm();
            test::check(theirs.has_value(), "minnorm: the oracle found a facet");
            // the oracle found a facet
            if (theirs) aux_below("minnorm", Z.minnorm().first.item(), *theirs);
        }
    });

    oracle::run("zonotope.intersectStrip", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : sizes) {
            const Zonotope Z = rnd.zonotope(n, m);
            const Tensor C = rnd.matrix(2, n), Lambda = rnd.matrix(n, 2), y = rnd.vector(2);
            const Tensor phi = Tensor::from_data({0.5, 1.5}, {2, 1});
            const lean::Zonotope theirs = lean::Zonotope(Z).intersect_strip(C, phi, y, Lambda);
            // phi and y (k); the oracle takes CORA's columns
            aux_hull_agrees(Z.intersect_strip(C, phi.squeeze(-1), y.squeeze(-1), Lambda), theirs);
        }
    });

    oracle::run("zonotope.isIntersecting", [&](oracle::Random &rnd) {
        for (const auto &[n, m] : {std::pair{2, 3}, {3, 5}}) {
            const Zonotope Z = rnd.zonotope(n, m), Z2 = rnd.zonotope(n, m);
            const Zonotope far =
                Zonotope((Z2.center().expand_dims(-1) * 40.0).squeeze(-1), Z2.generators());
            for (const Zonotope &S : {Z2, far}) {
                const lean::Certified r = lean::Zonotope(Z).is_intersecting(lean::Zonotope(S));
                // a proof of disjointness is never contradicted
                if (r.proved && !r.value)
                    test::check(!Z.is_intersecting(S, "exact", 0.0),
                                "isIntersecting: proved disjoint");
            }
        }
    });
    return test::finish("lean zonotope measures (bounds)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
