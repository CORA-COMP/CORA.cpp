// test_specification_unsafeSet - specification unsafeSet: a halfspace the set must not touch

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

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

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) { unsafeSet(b); });
    return test::finish("specification unsafeSet");
}
