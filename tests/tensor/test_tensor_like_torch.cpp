// test_tensor_like_torch - Library code makes its tensors on the backend of the ones it is given

#include "contDynamics/linearSys/linearSys.h"
#include "contSet/interval/interval.h"
#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "specification/specification.h"
#include "tensor/eigen.h"
#include "tensor/tensor.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::throws;

namespace {

/// Library code makes its tensors on the backend of the ones it was given, not on the current
/// one: an Eigen set stays on Eigen while libtorch is the default.
void tensors_stay_on_their_own_backend() {
    setBackend("eigen");
    const Zonotope Z(Tensor({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    const Interval I(Tensor({0.0, 0.0}), Tensor({1.0, 2.0}));
    const LinearSys sys(Tensor({{-0.2, 1.0}, {-1.0, -0.2}}));
    setBackend("torch");
    cora::Rng rng(1);
    check(!throws([&] { Z.randPoint(5, rng); }), "randPoint of an eigen zonotope, torch current");
    check(!throws([&] { Z.randPoint(5, rng, "extreme"); }), "extreme points of an eigen zonotope");
    check(!throws([&] { I.randPoint(5, rng); }), "randPoint of an eigen interval, torch current");
    check(!throws([&] { sys.simulate(Z.randPoint(5, rng), 0.1, 0.5); }),
          "simulate on eigen, torch current");
    check(!throws([&] { sys.reach(Z, 0.1, 0.5, 6); }), "reach on eigen, torch current");
    check(!throws([&] { Specification::safeSet(Tensor({1.0, 0.0}), 9.0); }), "a torch spec builds");
    setBackend("eigen");
}

} // namespace

int main() {
    tensors_stay_on_their_own_backend();
    setBackend("eigen");
    return test::finish("tensor like (libtorch)");
}
