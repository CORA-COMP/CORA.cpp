// test_specification_holds_torch - specifications on libtorch sets: single sets, no batch
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       29-September-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/torch.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    // A unit box at x1 = 2: its right edge is at 3.
    const Zonotope box(Tensor({2.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Polytope right(Tensor({{1.0, 0.0}}), Tensor({{3.0}}));
    const Polytope left(Tensor({{1.0, 0.0}}), Tensor({{2.5}}));

    check(Specification::safe_set(right).check(box), "safe set: the box touches x1 <= 3");
    check(!Specification::safe_set(left).check(box), "safe set: the box leaves x1 <= 2.5");

    // The unsafe halfspace x1 <= 1.5 touches the box (left edge at 1).
    check(!Specification::unsafe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({{1.5}}))).check(box),
          "unsafe set: touched");
    check(Specification::unsafe_set(Polytope(Tensor({{1.0, 0.0}}), Tensor({{0.5}}))).check(box),
          "unsafe set: clear");

    // A polytope: x1 <= 3 and x1 >= 1 (that is, -x1 <= -1) keeps the box.
    const Polytope band(Tensor({{1.0, 0.0}, {-1.0, 0.0}}), Tensor({{3.0}, {-1.0}}));
    check(Specification::safe_set(band).check(box), "a polytope of two rows");

    // The polytope operations take one set: a batch of three boxes is refused.
    const torch::Tensor c =
        torch::tensor({{{0.0}, {0.0}}, {{2.0}, {0.0}}, {{4.0}, {0.0}}}, torch::kDouble);
    const torch::Tensor G = torch::eye(2, torch::kDouble).expand({3, 2, 2}).contiguous();
    const Zonotope batch(from_torch(c), from_torch(G));
    check(test::raises([&] { Specification::safe_set(right).check(batch); }, Error::not_supported),
          "a batch of sets is refused");
    return test::finish("specification (libtorch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
