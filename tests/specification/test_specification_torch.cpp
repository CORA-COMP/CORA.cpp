// Specifications over a batch of sets: one answer per member.

#include "contSet/zonotope/zonotope.h"
#include "specification/specification.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora::ct;
using test::check;

int main() {
    set_backend("torch");
    // Three unit boxes at x1 = 0, 2 and 4: their right edges are at 1, 3 and 5.
    const torch::Tensor c = torch::tensor({{{0.0}, {0.0}}, {{2.0}, {0.0}}, {{4.0}, {0.0}}}, torch::kDouble);
    const torch::Tensor G = torch::eye(2, torch::kDouble).expand({3, 2, 2}).contiguous();
    const Zonotope batch(from_torch(c), from_torch(G));

    const Specification safe = Specification::safe_set(Tensor({1.0, 0.0}), 3.0);
    check(safe.holds(batch) == std::vector<bool>({true, true, false}), "safe set: one answer per member");
    check(!safe.check(batch), "safe set: check needs every member");

    const Specification clear = Specification::safe_set(Tensor({1.0, 0.0}), 5.0);
    check(clear.check(batch), "safe set: every member inside");

    // The unsafe halfspace x1 <= 1.5 touches the first two boxes (left edges at -1 and 1).
    const Specification unsafe = Specification::unsafe_set(Tensor({1.0, 0.0}), 1.5);
    check(unsafe.holds(batch) == std::vector<bool>({false, false, true}), "unsafe set: one answer per member");

    // A polytope: x1 <= 3 and x1 >= 1 (that is, -x1 <= -1) keeps only the middle box.
    const Specification band =
        Specification::safe_set({{Tensor({1.0, 0.0}), 3.0}, {Tensor({-1.0, 0.0}), -1.0}});
    check(band.holds(batch) == std::vector<bool>({false, true, false}), "a polytope over a batch");

    // A single set gives a single answer.
    const Zonotope one(Tensor({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(safe.holds(one).size() == 1, "a single set gives one answer");
    return test::finish("specification (libtorch)");
}
