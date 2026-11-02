// test_specification_safe_set - specification safe_set: one halfspace and polytopes

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/specification/specification.h"
#include "testing.h"

#include <stdexcept>

using namespace cora;
using test::check;
using test::close;

namespace {

/// The box `[1, 3] x [0, 2]` as a zonotope.
Zonotope box() { return {Tensor({2.0, 1.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})}; }

void safe_set(const std::string &b) {
    const Zonotope Z = box();
    check(Specification::safe_set(Tensor({1.0, 0.0}), 3.0).check(Z),
          b + ": x1 <= 3 touches: inside");
    check(!Specification::safe_set(Tensor({1.0, 0.0}), 2.9).check(Z), b + ": x1 <= 2.9 is left");
    check(Specification::safe_set(Tensor({-1.0, 0.0}), -1.0).check(Z), b + ": x1 >= 1 holds");
    check(!Specification::safe_set(Tensor({-1.0, 0.0}), -1.5).check(Z), b + ": x1 >= 1.5 is left");
    check(Specification::safe_set(Tensor({1.0, 1.0}), 5.0).check(Z), b + ": x1 + x2 <= 5 holds");
    check(!Specification::safe_set(Tensor({1.0, 1.0}), 4.9).check(Z),
          b + ": x1 + x2 <= 4.9 is left");
}

/// Several halfspaces are a polytope: all must hold.
void safe_polytope(const std::string &b) {
    const Zonotope Z = box();
    const Halfspace right{Tensor({1.0, 0.0}), 3.0}, top{Tensor({0.0, 1.0}), 2.0};
    check(Specification::safe_set({right, top}).check(Z), b + ": inside both");
    const Halfspace low_top{Tensor({0.0, 1.0}), 1.0};
    check(!Specification::safe_set({right, low_top}).check(Z), b + ": one of two fails");
    check(Specification::safe_set({right, top}).type() == SpecType::safe_set, b + ": the type");
    check(test::raises([] { Specification::safe_set(std::vector<Halfspace>{}); },
                       Error::wrong_input_in_constructor),
          b + ": no halfspace");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        safe_set(b);
        safe_polytope(b);
    });
    return test::finish("specification safe_set");
}
