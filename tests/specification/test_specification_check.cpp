// test_specification_check - specification check on any set and on a reachable set

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

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
        any_contset(b);
        on_a_reachable_set(b);
    });
    return test::finish("specification check");
}
