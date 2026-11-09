// test_specification_custom - specification custom: a predicate on the set
//
// CORA: specification/@specification/specification.m
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The predicate "the box stays below 2 in x1" on [1, 3] x [0, 2] and on a smaller set.
void custom(const std::string &b) {
    const Specification below =
        Specification::custom([](const ContSet &S) { return S.interval().sup().data()[0] <= 2.0; });
    check(below.type() == SpecType::custom && below.halfspaces().empty(), b + ": no halfspaces");
    const Zonotope wide(Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Zonotope narrow(Tensor({1.0, 1.0}), Tensor({{0.5, 0.0}, {0.0, 1.0}}));
    check(!below.check(wide) && below.check(narrow), b + ": the predicate decides");
    check(below.holds(narrow) == std::vector<bool>{true}, b + ": one answer for the set");
    check(!below.check(std::vector<Zonotope>{narrow, wide}), b + ": a sequence fails at one");
}

/// No predicate is refused, verify refuses custom specifications; checks off, the same.
void checks_and_their_switch() {
    check(test::raises([] { Specification::custom(nullptr); }, Error::wrong_input_in_constructor),
          "custom: no predicate");
    const Specification yes = Specification::custom([](const ContSet &) { return true; });
    const LinearSys sys(Tensor({{-1.0}}), Tensor({{1.0}}));
    const ReachProblem p{Zonotope(Tensor({1.0}), Tensor({{0.1}})),
                         Zonotope(Tensor({0.0}), Tensor({{0.0}})), 1.0};
    check(test::raises([&] { sys.verify(p, {yes}, alg::ReachAvoidZonotope{}); },
                       Error::not_supported),
          "verify: custom specifications are not decided");
    ChecksScope off(false);
    check(yes.check(Zonotope(Tensor({1.0}), Tensor({{0.1}}))), "custom: the same, checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { custom(b); });
    checks_and_their_switch();
    return test::finish("specification custom");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
