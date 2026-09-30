// test_linearSys_outputSet - linearSys outputSet: y = C x is exact on zonotopes, and without C the
// output is the state

#include "contDynamics/linearSys/linearSysTesting.h"
#include "contDynamics/linearSys/matlabReferenceInputs.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using namespace test::lin;

namespace {

struct Setup {
    Tensor A, B, C;
    Zonotope X0, U;
    Setup()
        : A(Tensor::fromData(matlab_reference::standardInputs().A, {3, 3})),
          B(Tensor::fromData(matlab_reference::standardInputs().B, {3, 2})),
          C(Tensor::fromData(matlab_reference::standardInputs().C, {2, 3})),
          X0(Tensor::fromData(matlab_reference::standardInputs().c0, {3, 1}),
             Tensor::fromData(matlab_reference::standardInputs().G0, {3, 3})),
          U(Tensor::fromData(matlab_reference::standardInputs().cU, {2, 1}),
            Tensor::fromData(matlab_reference::standardInputs().GU, {2, 3})) {}
};

/// The output set is C Z: its center is C c, its generators C G, and its support along d is the
/// support of the state set along C'd.
void maps_every_set_by_C(const std::string &b) {
    const Setup in;
    const LinearSys sys(in.A, in.B, in.C);
    const Reach R = sys.reach(in.X0, in.U, 0.2, 1.0, 6);
    const Reach Y = sys.outputSet(R);
    check(Y.timeInt.size() == R.timeInt.size() && Y.timePoint.size() == R.timePoint.size(),
          b + ": the number of sets");
    const std::vector<double> d{0.7, -1.1};
    const Tensor Ctd = in.C.transpose().matmul(test::column(d));
    for (std::size_t k = 0; k < R.timeInt.size(); ++k) {
        check(close(Y.timeInt[k].c, in.C.matmul(R.timeInt[k].c)), b + ": the center of a step");
        check(close(Y.timeInt[k].supportFunc(test::column(d)), R.timeInt[k].supportFunc(Ctd)),
              b + ": the support of a step along C'd");
    }
    for (std::size_t k = 0; k < R.timePoint.size(); ++k)
        check(close(Y.timePoint[k].G, in.C.matmul(R.timePoint[k].G)), b + ": a time point");
    check(Y.timeInt[0].dim() == 2, b + ": the output has p dimensions");
}

void is_the_state_without_C(const std::string &b) {
    const Setup in;
    const LinearSys sys(in.A, in.B);
    const Reach R = sys.reach(in.X0, in.U, 0.2, 0.6, 6);
    const Reach Y = sys.outputSet(R);
    check(close(Y.timeInt[1].c, R.timeInt[1].c) && close(Y.timePoint[2].G, R.timePoint[2].G),
          b + ": without C the output is the state");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        maps_every_set_by_C(b);
        is_the_state_without_C(b);
    });
    return test::finish("linearSys outputSet");
}
