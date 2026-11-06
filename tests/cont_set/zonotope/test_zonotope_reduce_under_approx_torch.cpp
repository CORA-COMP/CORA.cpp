// test_zonotope_reduce_under_approx_torch - zonotope reduce_under_approx: gradients through torch
//
// The generators to merge are chosen on the values; given the choice the result is linear in
// the generators ("kochdumper", "raghuraman") or a smooth function of them ("yang").
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zonotope_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(56);
    const auto input = test::random_zonotope(2, 6);
    for (const char *method : {"kochdumper", "raghuraman", "yang"})
        test::gradcheck(
            std::string("reduce_under_approx ") + method,
            [method](const Inputs &x) {
                return test::parts(Zonotope(x[0], x[1]).reduce_under_approx(method, 1.5));
            },
            input);
    return test::finish("zonotope reduce_under_approx (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
