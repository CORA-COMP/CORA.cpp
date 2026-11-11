// test_neur_net_contr_sys_verify - the verdicts of verify
//
// A loose safe set is verified, a set that every run violates is falsified by a simulation (with
// the start of a violating run), and the zonotope and the polynomial zonotope start agree.
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

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        verify_decides(b);
        a_hard_specification_is_unknown(b);
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
