// test_tensor_like_torch - Library code makes its tensors on the backend of the ones it is given
//
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/eigen.h"
#include "cora/global/backend/torch.h"
#include "cora/global/rng.h"
#include "cora/global/tensor/tensor.h"
#include "cora/specification/specification.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Library code makes its tensors on the backend of the ones it was given, not on the current
/// one: an Eigen set stays on Eigen while libtorch is the default.
void tensors_stay_on_their_own_backend() {
    set_default_backend("eigen");
    const Zonotope Z(Tensor({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Interval I(Tensor({0.0, 0.0}), Tensor({1.0, 2.0}));
    const LinearSys sys(Tensor({{-0.2, 1.0}, {-1.0, -0.2}}));
    set_default_backend("torch");
    cora::Rng rng(1);
    check(!throws([&] { Z.rand_point(5, rng); }), "rand_point of an eigen zonotope, torch current");
    check(!throws([&] { Z.rand_point(5, rng, "extreme"); }), "extreme points of an eigen zonotope");
    check(!throws([&] { I.rand_point(5, rng); }), "rand_point of an eigen interval, torch current");
    check(!throws([&] {
        sys.simulate({.initial_state = Z.rand_point(5, rng), .t_final = 0.5},
                     alg::Exact{.time_step = 0.1});
    }),
          "simulate on eigen, torch current");
    check(!throws([&] {
        sys.reach({.initial_set = Z, .t_final = 0.5},
                  alg::Standard{.time_step = 0.1, .taylor_terms = 6});
    }),
          "reach on eigen, torch current");
    check(!throws([&] { Specification::safe_set(Tensor({1.0, 0.0}), 9.0); }),
          "a torch spec builds");
    set_default_backend("eigen");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    tensors_stay_on_their_own_backend();
    set_default_backend("eigen");
    return test::finish("tensor like (libtorch)");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
