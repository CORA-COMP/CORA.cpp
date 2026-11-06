// test_zonotope_split_torch - zonotope split, split_first_gen and split_longest_gen: gradients
//
// The pieces are linear in the center and the generators once the generator or the coordinate
// is chosen (on the values): the gradient of the choice made.
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

Inputs pieces(const std::pair<Zonotope, Zonotope> &p) {
    Inputs out = test::parts(p.first);
    for (const Tensor &t : test::parts(p.second)) out.push_back(t);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(55);
    const auto input = test::random_zonotope(3, 4);
    test::gradcheck(
        "split by a coordinate",
        [](const Inputs &x) { return pieces(Zonotope(x[0], x[1]).split(1)); }, input);
    test::gradcheck(
        "split along a generator",
        [](const Inputs &x) { return pieces(Zonotope(x[0], x[1]).split(2, 4)); }, input);
    test::gradcheck(
        "split by a direction",
        [](const Inputs &x) {
            return pieces(Zonotope(x[0], x[1])
                              .split(cora::from_torch(
                                  torch::tensor({{1.0}, {2.0}, {-1.0}}, torch::kDouble))));
        },
        input);
    test::gradcheck(
        "split_first_gen",
        [](const Inputs &x) { return pieces(Zonotope(x[0], x[1]).split_first_gen()); }, input);
    test::gradcheck(
        "split_longest_gen",
        [](const Inputs &x) { return pieces(Zonotope(x[0], x[1]).split_longest_gen()); }, input);
    return test::finish("zonotope split (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
