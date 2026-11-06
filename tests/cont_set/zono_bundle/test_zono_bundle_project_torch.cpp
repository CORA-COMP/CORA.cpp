// test_zono_bundle_project_torch - ZonoBundle project: gradients through torch
//
// A gather of the rows of every member: differentiable.
//
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_gradients.h"

using namespace cora;
using test::Inputs;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    torch::manual_seed(73);
    // the dimensions 2 and 0 of a bundle in three dimensions
    test::gradcheck(
        "project",
        [](const Inputs &x) {
            const ZonoBundle projected = Set(test::bundle(x, 0, 2))->project({2, 0}).as<ZonoBundle>();
            return test::parts(projected);
        },
        test::random_bundle_parts(2, 3, 2));
    return test::finish("zono_bundle project (torch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
