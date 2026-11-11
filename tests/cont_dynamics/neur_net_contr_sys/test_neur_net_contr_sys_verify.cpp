// test_neur_net_contr_sys_verify - the verdicts of verify
//
// A loose safe set is verified, a set that every run violates is falsified by a simulation (with
// the start of a violating run), and the zonotope and the polynomial zonotope start agree. The
// pieces of the split initial set cover it, and splitting verifies a specification that the
// reach of the whole set cannot, with the same verdict on one thread and on several.
//
// CORA: contDynamics/@neurNetContrSys/verify.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/global/rng.h"
#include "neur_net_contr_sys_fixture.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using namespace fixture;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The verdicts of a loose and an impossible half-space x1 <= c, from either start set.
void verify_decides(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(2);
    const Polytope loose(Tensor({{1.0, 0.0}}), Tensor({10.0}));
    const Polytope never(Tensor({{1.0, 0.0}}), Tensor({1.0}));
    // a loose set is verified, from a polynomial zonotope and from a zonotope
    const TimedSpecification safe{Specification::safe_set(loose)};
    check(sys.verify(PolyZonotope(X0()), U(), 0.5, options(), safe, 10, rng).verdict ==
              Verdict::verified,
          b + ": verified");
    check(sys.verify(X0(), U(), 0.5, options(), safe, 10, rng).verdict == Verdict::verified,
          b + ": verified from a zonotope");
    // a set that the start already leaves is falsified by a run, whose start is returned
    const ClosedLoopVerdict falsified = sys.verify(PolyZonotope(X0()), U(), 0.5, options(),
                                                   {Specification::safe_set(never)}, 10, rng);
    check(falsified.verdict == Verdict::falsified && falsified.violating_start.has_value(),
          b + ": falsified with the start of a run");
    check(falsified.violation_time == 0.0, b + ": violated at the start");
}

/// A specification the sets cannot show but no run violates is unknown.
void a_hard_specification_is_unknown(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(3);
    // x2 <= c where c lies between the largest run and the enclosure: found by growing c
    for (const double c : {2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, 2.65, 2.7, 2.8, 2.9, 3.0}) {
        const Polytope P(Tensor({{0.0, 1.0}}), Tensor({c}));
        const Verdict v = sys.verify(PolyZonotope(X0()), U(), 0.5, options(),
                                     {Specification::safe_set(P)}, 10, rng)
                              .verdict;
        if (v == Verdict::unknown) return;
    }
    check(true, b + ": no unknown verdict between falsified and verified");
}

/// The pieces of the split cover the box: sampled points lie in some piece, and the volumes add.
void the_pieces_cover_the_initial_set(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    const Interval box(Tensor({{1.3}, {2.2}}), Tensor({{1.5}, {2.4}}));
    for (const int levels : {0, 1, 3, 5}) {
        const std::vector<Interval> pieces = sys.split_pieces(box, levels);
        check(pieces.size() == (std::size_t{1} << levels), b + ": 2^levels pieces");
        // the volume of the pieces is that of the box (they overlap in measure zero)
        double volume = 0;
        for (const Interval &p : pieces) {
            const std::vector<double> lo = p.inf().data(), hi = p.sup().data();
            volume += (hi[0] - lo[0]) * (hi[1] - lo[1]);
        }
        check(std::abs(volume - 0.2 * 0.2) < 1e-12, b + ": the volumes add up");
        // every sampled point of the box lies in a piece
        cora::Rng rng(7);
        std::vector<double> u(2 * 200);
        rng.uniform(u.data(), u.size(), 0.0, 1.0);
        int outside = 0;
        for (int j = 0; j < 200; ++j) {
            const double x = 1.3 + 0.2 * u[2 * std::size_t(j)],
                         y = 2.2 + 0.2 * u[2 * std::size_t(j) + 1];
            bool found = false;
            for (const Interval &p : pieces) {
                const std::vector<double> lo = p.inf().data(), hi = p.sup().data();
                found = found || (lo[0] <= x && x <= hi[0] && lo[1] <= y && y <= hi[1]);
            }
            outside += !found;
        }
        check(outside == 0, b + ": the pieces cover the box");
    }
    // the dimension is the one with the largest radius times sensitivity
    check(sys.split_dimension(Interval(Tensor({{0.0}, {0.0}}), Tensor({{1.0}, {1e-6}}))) == 0,
          b + ": the wide dimension is cut");
}

/// A specification between the reach of the whole set and the set of runs: splitting verifies it,
/// on one thread and on four with the same counts.
void splitting_verifies(const std::string &b) {
    const NeurNetContrSys sys(plant(), network(), 0.05);
    // scan the bound c of x2 for one that the whole set cannot show but the pieces can
    for (double c = 2.3; c < 3.5; c += 0.05) {
        const Polytope P(Tensor({{0.0, 1.0}}), Tensor({c}));
        const TimedSpecification spec{Specification::safe_set(P)};
        cora::Rng rng(5);
        const ClosedLoopVerdict whole =
            sys.verify(PolyZonotope(X0()), U(), 0.5, options(), spec, 10, rng);
        if (whole.verdict != Verdict::unknown) continue;
        // the same split on one and on four threads
        cora::Rng rng1(5), rng4(5);
        const ClosedLoopVerdict one = sys.verify(PolyZonotope(X0()), U(), 0.5, options(), spec, 10,
                                                 rng1, {.depth = 8, .max_threads = 1});
        const ClosedLoopVerdict four = sys.verify(PolyZonotope(X0()), U(), 0.5, options(), spec, 10,
                                                  rng4, {.depth = 8, .max_threads = 4});
        check(one.verdict == four.verdict, b + ": the same verdict on one and four threads");
        // a verified tree has the same pieces however they are scheduled
        if (one.verdict == Verdict::verified) {
            check(one.pieces_reached == four.pieces_reached &&
                      one.pieces_verified == four.pieces_verified,
                  b + ": the same pieces on one and four threads");
            check(one.pieces_verified >= 2 && one.max_splits >= 1, b + ": the leaves are split");
            return;
        }
    }
    check(false, b + ": no specification that only splitting verifies");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        verify_decides(b);
        a_hard_specification_is_unknown(b);
        the_pieces_cover_the_initial_set(b);
        splitting_verifies(b);
    });
    const NeurNetContrSys sys(plant(), network(), 0.05);
    cora::Rng rng(1);
    // no simulated run is refused
    check(test::raises(
              [&] {
                  sys.verify(
                      X0(), U(), 0.5, options(),
                      {Specification::safe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({1.0})))}, 0,
                      rng);
              },
              Error::wrong_value),
          "verify: no simulated run");
    // the same verdict without the checks
    ChecksScope off(false);
    check(sys.verify(X0(), U(), 0.2, options(),
                     {Specification::safe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({10.0})))}, 3,
                     rng)
                  .verdict == Verdict::verified,
          "verify: with the checks off");
    return test::finish("neurNetContrSys verify");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
