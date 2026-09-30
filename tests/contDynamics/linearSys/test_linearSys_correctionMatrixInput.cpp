// test_linearSys_correctionMatrixInput - the correction matrix G(A, dt, taylorTerms) of linearSys
// against MATLAB CORA's `taylorMatrices`, and the properties it has whatever the numbers.

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/linearSys/matlabReferenceInputs.h"
#include "contDynamics/linearSys/linearSysTesting.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using namespace test::lin;
using test::check;
using test::close;

namespace {

double max_abs(const Tensor &t) {
    double m = 0.0;
    for (const double v : t.data()) m = std::max(m, std::abs(v));
    return m;
}

void matches_matlab_cora(const std::string &b) {
    const matlab_reference::InputSystem &s = matlab_reference::standardInputs();
    const Interval G = LinearSys(Tensor::fromData(s.A, {3, 3}), Tensor::fromData(s.B, {3, 2}))
                           .correctionMatrixInput(s.timeStep, s.taylorTerms);
    check(G.inf.shape() == std::vector<int64_t>({3, 3}), b + ": shape");
    check(close(G.inf, s.G_inf, 1e-12), b + ": the lower bounds differ from MATLAB CORA");
    check(close(G.sup, s.G_sup, 1e-12), b + ": the upper bounds differ from MATLAB CORA");
}

void is_zero_for_a_zero_system(const std::string &b) {
    const Interval G = LinearSys(Tensor::zeros({3, 3})).correctionMatrixInput(0.1, 6);
    check(max_abs(G.inf) == 0.0 && max_abs(G.sup) == 0.0, b + ": G of A = 0 is zero");
}

/// For A = -a the integral of e^{-a s} u is 1 - e^{-a dt} over a; the chord of the trajectory
/// (time 0 to dt) lies above it by a bounded amount that G u must contain.
void encloses_the_curvature_of_a_scalar_system(const std::string &b) {
    const double a = 1.3, dt = 0.4;
    const Interval G = LinearSys(Tensor({{-a}})).correctionMatrixInput(dt, 10);
    // x(t) = (1 - e^{-a t}) / a for u = 1 and x(0) = 0; the chord is t x(dt) / dt.
    double curvature = 0.0;
    for (int i = 0; i <= 100; ++i) {
        const double t = dt * i / 100.0;
        const double x = (1 - std::exp(-a * t)) / a, chord = t / dt * (1 - std::exp(-a * dt)) / a;
        curvature = std::max(curvature, std::abs(x - chord));
    }
    const double lo = G.inf.data()[0], hi = G.sup.data()[0];
    check(lo <= 0.0 && hi >= 0.0, b + ": G contains zero");
    check(hi >= curvature - 1e-12 || -lo >= curvature - 1e-12,
          b + ": G does not reach the curvature " + std::to_string(curvature));
}

void shrinks_with_the_time_step(const std::string &b) {
    const matlab_reference::InputSystem &s = matlab_reference::standardInputs();
    const LinearSys sys(Tensor::fromData(s.A, {3, 3}));
    double previous = 1e9;
    for (const double dt : {0.4, 0.2, 0.1, 0.05, 0.025}) {
        const double size = max_abs(sys.correctionMatrixInput(dt, 8).rad());
        check(size < previous, b + ": G did not shrink at dt = " + std::to_string(dt));
        previous = size;
    }
}

void is_tighter_with_more_terms(const std::string &b) {
    const matlab_reference::InputSystem &s = matlab_reference::standardInputs();
    const LinearSys sys(Tensor::fromData(s.A, {3, 3}));
    check(max_abs(sys.correctionMatrixInput(0.2, 10).rad()) <=
              max_abs(sys.correctionMatrixInput(0.2, 3).rad()),
          b + ": more Taylor terms widened G");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        matches_matlab_cora(b);
        is_zero_for_a_zero_system(b);
        encloses_the_curvature_of_a_scalar_system(b);
        shrinks_with_the_time_step(b);
        is_tighter_with_more_terms(b);
    });
    return test::finish("linearSys correctionMatrixInput");
}
