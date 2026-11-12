// test_linear_sys_krylov_windows - the restart windows of a Krylov trajectory enclose the solution
//
// A chain of 40 coupled states with a growth rate mu_T of the Hessenberg matrix above zero (the
// bound e^{t mu} of one subspace is useless over t = 40): the trajectory is cut into windows, and
// at the end of every window the approximation |s| V e^{H t} e_1 is within the window's error
// bound of the true e^{tA} v (the matrix exponential), the windows add up to the horizon, and a
// tolerance that no window can meet is refused.
//
// CORA.cpp only: CORA has no restart; its single subspace runs to the final time.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_krylov.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A chain x_i' = -x_i + 0.6 (x_{i-1} - x_{i+1}) + 0.1 x_{i+1}: stable, far from normal.
Tensor chain(int64_t n) {
    std::vector<double> A(size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) {
        A[size_t(i * n + i)] = -1.0;
        if (i > 0) A[size_t(i * n + i - 1)] = 0.6;
        if (i + 1 < n) A[size_t(i * n + i + 1)] = -0.5;
    }
    return Tensor::from_data(A, {n, n});
}

/// The 2-norm of the difference of two columns.
double distance(const Tensor &a, const Tensor &b) { return (a - b).square().sum().sqrt().item(); }

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const int64_t n = 40;
    const Tensor A = chain(n);
    std::vector<double> v(n);
    for (int64_t i = 0; i < n; ++i) v[size_t(i)] = std::sin(1.0 + i);
    const Tensor v0 = Tensor::from_data(v, {n, 1});
    const HostMatVec M = priv_state_matvec(LinearSys(A));
    const double dt = 0.5;
    const int64_t steps = 80;

    // the windows with a tolerance that one subspace cannot meet over the horizon
    KrylovWindows windows(M, v, dt, steps, 1e-8, 5, 5, v0);
    int64_t covered = 0, count = 0;
    double worst = 0;
    while (!windows.done()) {
        const KrylovWindow w = windows.next();
        covered += w.steps;
        ++count;
        // the approximation at the end of the window against the true solution there
        const Tensor y = (w.H * (double(w.steps) * dt)).expm().slice(-1, 0, 1);
        const Tensor approx = w.V.matmul(y) * w.norm;
        const Tensor exact = (A * (double(covered) * dt)).expm().matmul(v0);
        worst = std::max(worst, distance(approx, exact) / w.error);
    }
    check(covered == steps, "the windows cover the horizon");
    check(count >= 2, "more than one window: " + std::to_string(count));
    check(worst <= 1.0, "the error is within the bound of the window: " + std::to_string(worst));

    // a short horizon and a loose tolerance: one window
    KrylovWindows easy(M, v, dt, 8, 1e-2, 5, 5, v0);
    const KrylovWindow whole = easy.next();
    check(easy.done() && whole.steps == 8, "a short horizon needs one window");

    // a tolerance below the rounding of the start vector is refused
    KrylovWindows impossible(M, v, dt, steps, 1e-300, 5, 5, v0);
    check(test::raises([&] { impossible.next(); }, Error::not_converged),
          "an impossible tolerance");
    return test::finish("linear_sys krylov windows");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
