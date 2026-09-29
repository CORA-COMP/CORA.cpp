// Specifications on every backend: safe and unsafe halfspaces against zonotopes and boxes,
// and against a reachable set.

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "testing.h"

#include <stdexcept>

using namespace cora::ct;
using test::check;

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

void safeSet(const std::string &b) {
    const Zonotope Z = box();
    check(Specification::safeSet(Tensor({1.0, 0.0}), 3.0).check(Z), b + ": x1 <= 3 touches: inside");
    check(!Specification::safeSet(Tensor({1.0, 0.0}), 2.9).check(Z), b + ": x1 <= 2.9 is left");
    check(Specification::safeSet(Tensor({-1.0, 0.0}), -1.0).check(Z), b + ": x1 >= 1 holds");
    check(!Specification::safeSet(Tensor({-1.0, 0.0}), -1.5).check(Z), b + ": x1 >= 1.5 is left");
    check(Specification::safeSet(Tensor({1.0, 1.0}), 5.0).check(Z), b + ": x1 + x2 <= 5 holds");
    check(!Specification::safeSet(Tensor({1.0, 1.0}), 4.9).check(Z), b + ": x1 + x2 <= 4.9 is left");
}

/// Several halfspaces are a polytope: all must hold.
void safe_polytope(const std::string &b) {
    const Zonotope Z = box();
    const Halfspace right{Tensor({1.0, 0.0}), 3.0}, top{Tensor({0.0, 1.0}), 2.0};
    check(Specification::safeSet({right, top}).check(Z), b + ": inside both");
    const Halfspace low_top{Tensor({0.0, 1.0}), 1.0};
    check(!Specification::safeSet({right, low_top}).check(Z), b + ": one of two fails");
    check(Specification::safeSet({right, top}).type() == SpecType::SafeSet, b + ": the type");
    bool threw = false;
    try {
        Specification::safeSet(std::vector<Halfspace>{});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, b + ": no halfspace");
}

/// The unsafe halfspace {x | aᵀx <= b} must not touch the set.
void unsafeSet(const std::string &b) {
    const Zonotope Z = box();
    check(Specification::unsafeSet(Tensor({1.0, 0.0}), 0.5).check(Z), b + ": x1 <= 0.5 is clear");
    check(!Specification::unsafeSet(Tensor({1.0, 0.0}), 1.0).check(Z), b + ": touching counts");
    check(!Specification::unsafeSet(Tensor({1.0, 0.0}), 2.0).check(Z), b + ": overlapping");
    check(!Specification::unsafeSet(Tensor({1.0, 0.0}), 9.0).check(Z), b + ": containing");
    check(Specification::unsafeSet(Tensor({0.0, -1.0}), -2.5).check(Z), b + ": x2 >= 2.5 is clear");
    check(Specification::unsafeSet(Tensor({1.0, 0.0}), 0.5).type() == SpecType::UnsafeSet,
          b + ": the type");
}

/// Any set answers, since a halfspace needs only the support function.
void any_contset(const std::string &b) {
    const Interval I(Tensor({1.0, 0.0}), Tensor({3.0, 2.0}));
    check(Specification::safeSet(Tensor({1.0, 0.0}), 3.0).check(I), b + ": an interval, safe");
    check(!Specification::safeSet(Tensor({1.0, 0.0}), 2.0).check(I), b + ": an interval, left");
    check(Specification::unsafeSet(Tensor({1.0, 0.0}), 0.5).check(I), b + ": an interval, clear");
    const ContSet &generic = I;
    check(Specification::safeSet(Tensor({0.0, 1.0}), 2.0).holds(generic).size() == 1,
          b + ": one answer for a single set");
}

/// x' = -x from [1, 2]: the set shrinks towards 0, but the first enclosure still reaches 2.
void on_a_reachable_set(const std::string &b) {
    const LinearSys sys(Tensor({{-1.0}}));
    const Zonotope X0(Tensor({1.5}), Tensor({{0.5}}));
    const Reach R = sys.reach(X0, 0.1, 1.0, 8);

    check(Specification::safeSet(Tensor({1.0}), 2.1).check(R.timeInt), b + ": stays below 2.1");
    check(Specification::safeSet(Tensor({1.0}), 2.1).firstViolation(R.timeInt) == -1,
          b + ": no violation");
    check(!Specification::safeSet(Tensor({1.0}), 1.9).check(R.timeInt), b + ": exceeds 1.9");
    check(Specification::safeSet(Tensor({1.0}), 1.9).firstViolation(R.timeInt) == 0,
          b + ": exceeds 1.9 in the first step");

    // The lowest point decays as e^{-t}, so it passes 0.5 at t = 0.69; the first enclosure to
    // touch that level is reported, and the ones before it are clear.
    const Specification reaches_half = Specification::unsafeSet(Tensor({1.0}), 0.5);
    const int hit = reaches_half.firstViolation(R.timeInt);
    check(hit >= 4 && hit <= 7, b + ": the flow reaches x <= 0.5 at step " + std::to_string(hit));
    check(reaches_half.check(std::vector<Zonotope>(R.timeInt.begin(), R.timeInt.begin() + hit)),
          b + ": clear before that step");
    check(Specification::unsafeSet(Tensor({1.0}), -1.0).check(R.timeInt), b + ": never below -1");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        safeSet(b);
        safe_polytope(b);
        unsafeSet(b);
        any_contset(b);
        on_a_reachable_set(b);
    });
    return test::finish("specification");
}
