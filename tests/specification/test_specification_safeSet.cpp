// test_specification_safeSet - specification safeSet: one halfspace and polytopes

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "testing.h"

#include <stdexcept>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

void safeSet(const std::string &b) {
    const Zonotope Z = box();
    check(Specification::safeSet(Tensor({1.0, 0.0}), 3.0).check(Z),
          b + ": x1 <= 3 touches: inside");
    check(!Specification::safeSet(Tensor({1.0, 0.0}), 2.9).check(Z), b + ": x1 <= 2.9 is left");
    check(Specification::safeSet(Tensor({-1.0, 0.0}), -1.0).check(Z), b + ": x1 >= 1 holds");
    check(!Specification::safeSet(Tensor({-1.0, 0.0}), -1.5).check(Z), b + ": x1 >= 1.5 is left");
    check(Specification::safeSet(Tensor({1.0, 1.0}), 5.0).check(Z), b + ": x1 + x2 <= 5 holds");
    check(!Specification::safeSet(Tensor({1.0, 1.0}), 4.9).check(Z),
          b + ": x1 + x2 <= 4.9 is left");
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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        safeSet(b);
        safe_polytope(b);
    });
    return test::finish("specification safeSet");
}
