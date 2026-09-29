// test_linearSys_correctionMatrixState - the correction matrix F(A, Δt, η) of linearSys against
// MATLAB CORA's `taylorMatrices`, and the properties it has whatever the numbers.

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/linearSys/matlabReference.h"
#include "testing.h"

#include <cmath>

using namespace cora::ct;
using matlab_reference::System;
using test::check;
using test::close;

namespace {

LinearSys system_of(const System &s) { return LinearSys(Tensor::fromData(s.A, {s.n, s.n})); }

double max_abs(const Tensor &t) {
    double m = 0.0;
    for (const double v : t.data()) m = std::max(m, std::abs(v));
    return m;
}

void matches_matlab_cora(const std::string &b) {
    for (const System *s : {&matlab_reference::oscillator(), &matlab_reference::three_dimensional(),
                            &matlab_reference::scalar()}) {
        const Interval F = system_of(*s).correctionMatrixState(s->timeStep, s->taylorTerms);
        const std::string what = b + ": n=" + std::to_string(s->n);
        check(F.inf.shape() == std::vector<int64_t>({s->n, s->n}), what + ": shape");
        check(close(F.inf, s->F_inf, 1e-13), what + ": the lower bounds differ from MATLAB CORA");
        check(close(F.sup, s->F_sup, 1e-13), what + ": the upper bounds differ from MATLAB CORA");
    }
}

void is_zero_for_a_zero_system(const std::string &b) {
    const Interval F = LinearSys(Tensor::zeros({3, 3})).correctionMatrixState(0.1, 6);
    check(max_abs(F.center()) == 0.0 && max_abs(F.rad()) == 0.0, b + ": F of A = 0 is zero");
}

void is_an_interval(const std::string &b) {
    const System &s = matlab_reference::three_dimensional();
    const Interval F = system_of(s).correctionMatrixState(s.timeStep, s.taylorTerms);
    bool ordered = true;
    for (const double r : F.rad().data()) ordered &= r >= 0.0;
    check(ordered, b + ": the radius is not negative");
}

/// The curvature of a shorter step is smaller: F vanishes with the time step.
void shrinks_with_the_time_step(const std::string &b) {
    const System &s = matlab_reference::oscillator();
    double previous = 1e9;
    for (const double dt : {0.4, 0.2, 0.1, 0.05, 0.025}) {
        const double size = max_abs(system_of(s).correctionMatrixState(dt, 8).rad());
        check(size < previous, b + ": F did not shrink at Δt = " + std::to_string(dt));
        previous = size;
    }
}

/// The remainder of the Taylor series is smaller, and F with it, when more terms are kept.
void is_tighter_with_more_terms(const std::string &b) {
    const System &s = matlab_reference::three_dimensional();
    const LinearSys sys = system_of(s);
    check(max_abs(sys.correctionMatrixState(0.2, 10).rad()) <=
              max_abs(sys.correctionMatrixState(0.2, 3).rad()),
          b + ": more Taylor terms widened F");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        matches_matlab_cora(b);
        is_zero_for_a_zero_system(b);
        is_an_interval(b);
        shrinks_with_the_time_step(b);
        is_tighter_with_more_terms(b);
    });
    return test::finish("linearSys correctionMatrixState");
}
