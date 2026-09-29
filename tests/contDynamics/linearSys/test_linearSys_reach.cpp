// linearSys.reach on every backend: against MATLAB CORA, against the true trajectories, and
// against properties the algorithms must have whatever the numbers.

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/linearSys/matlabReference.h"
#include "global/rng.h"
#include "testing.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>

using namespace cora::ct;
using test::check;
using test::close;
using matlab_reference::System;

namespace {

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

std::string name(Algorithm a) { return a == Algorithm::Standard ? "standard" : "wrapping-free"; }

Zonotope set_of(const System &s) {
    return {Tensor::fromData(s.c, {s.n, 1}), Tensor::fromData(s.G, {s.n, s.m})};
}

Tensor system_matrix(const System &s) { return Tensor::fromData(s.A, {s.n, s.n}); }

double support(const Zonotope &Z, const Eigen::VectorXd &d) {
    return Z.supportFunc(Tensor::fromData({d.data(), d.data() + d.size()}, {d.size(), 1}))
        .data()[0];
}

/// A host matrix as a tensor on the current backend.
Tensor tensor_of(const Eigen::MatrixXd &M) {
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> rows = M;
    return Tensor::fromData({rows.data(), rows.data() + rows.size()}, {M.rows(), M.cols()});
}

Eigen::MatrixXd host_matrix(const std::vector<double> &row_major, int rows, int cols) {
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(row_major.data(), rows,
                                                                             cols);
}

/// Against MATLAB CORA. Without inputs its two algorithms give the wrapping-free sets; our
/// standard algorithm agrees on the first step, where `F X0` is the same.
void matches_matlab_cora(const std::string &b) {
    for (const System *s : {&matlab_reference::oscillator(), &matlab_reference::three_dimensional(),
                            &matlab_reference::scalar()}) {
        const std::size_t steps = s->timeInt.size();
        const int n = s->n;
        for (const Algorithm algorithm : kAlgorithms) {
            const Reach r = LinearSys(system_matrix(*s))
                                .reach(set_of(*s), s->timeStep, s->timeStep * steps,
                                       s->taylorTerms, algorithm);
            const std::string what = b + ": " + name(algorithm) + " n=" + std::to_string(n);
            check(r.timeInt.size() == steps && r.timePoint.size() == steps + 1,
                  what + ": the number of steps");
            double ti_err = 0.0, tp_err = 0.0;
            for (std::size_t k = 0; k < steps; ++k) {
                // Only the first step of the standard algorithm is CORA's.
                if (algorithm == Algorithm::Standard && k > 0) break;
                for (std::size_t i = 0; i < s->dirs.size() / n; ++i) {
                    const Eigen::VectorXd d = host_matrix(s->dirs, s->dirs.size() / n, n).row(i);
                    ti_err = std::max(ti_err, std::abs(support(r.timeInt[k], d) - s->timeInt[k][i]));
                }
            }
            for (std::size_t k = 0; k < s->timePoint.size(); ++k)
                for (std::size_t i = 0; i < s->dirs.size() / n; ++i) {
                    const Eigen::VectorXd d = host_matrix(s->dirs, s->dirs.size() / n, n).row(i);
                    tp_err = std::max(tp_err, std::abs(support(r.timePoint[k], d) - s->timePoint[k][i]));
                }
            check(ti_err < 1e-12, what + ": enclosures differ from MATLAB CORA by " + std::to_string(ti_err));
            check(tp_err < 1e-12, what + ": time points differ from MATLAB CORA by " + std::to_string(tp_err));
        }
    }
}

struct Case {
    Eigen::MatrixXd A;
    double dt;
    std::string what;
};

/// Points of the initial set pushed along the flow, at times inside every step, must lie
/// under the support function of the step's enclosure in every sampled direction. The fast
/// rotation turns a long way per step, so the flow bulges far out of the chord between the
/// time points and only the curvature enlargement keeps it inside.
void encloses_the_trajectories(const std::string &b) {
    cora::Rng rng(3);
    const int n = 3, m = 4;
    Eigen::MatrixXd gentle(n, n), rotation = Eigen::MatrixXd::Zero(n, n);
    gentle << -0.3, 1.0, 0.0, -1.0, -0.2, 0.4, 0.0, -0.5, -0.1;
    rotation(0, 1) = 3.0;
    rotation(1, 0) = -3.0;
    rotation(2, 2) = -0.5;

    for (const Case &tc : {Case{gentle, 0.1, "gentle"}, Case{rotation, 0.3, "fast rotation"}}) {
        const Zonotope X0 = Zonotope::generateRandom(n, m, rng);
        const Eigen::MatrixXd c0 = host_matrix(X0.c.data(), n, 1), g0 = host_matrix(X0.G.data(), n, m);
        const Tensor A = tensor_of(tc.A);
        for (const Algorithm algorithm : kAlgorithms) {
            const Reach r = LinearSys(A).reach(X0, tc.dt, 2.0, 10, algorithm);
            const std::string what = b + ": " + tc.what + ": " + name(algorithm);

            const int samples = 300;
            Eigen::MatrixXd beta(m, samples), dirs(n, samples), times(1, samples);
            rng.uniform(beta.data(), beta.size(), -1.0, 1.0);
            rng.normal(dirs.data(), dirs.size(), 1.0);
            rng.uniform(times.data(), times.size(), 0.0, 1.0);
            double worst = 1e9;
            for (std::size_t k = 0; k < r.timeInt.size(); ++k)
                for (int i = 0; i < samples; ++i) {
                    const Eigen::MatrixXd flow = (tc.A * (double(k) + times(i)) * tc.dt).exp();
                    const Eigen::VectorXd x = flow * (c0 + g0 * beta.col(i));
                    const Eigen::VectorXd d = dirs.col(i);
                    worst = std::min(worst, (support(r.timeInt[k], d) - d.dot(x)) / d.norm());
                }
            check(worst >= -1e-9, what + ": a trajectory left its enclosure by " + std::to_string(-worst));
        }
    }
}

/// The time-point sets are `e^{A k Δt} X0` exactly, so their support along `d` is the
/// initial set's along `e^{A k Δt}ᵀ d`, computed here on the host.
void time_points_are_exact(const std::string &b) {
    const System &s = matlab_reference::three_dimensional();
    const Eigen::MatrixXd A = host_matrix(s.A, 3, 3), G = host_matrix(s.G, 3, 3);
    const Eigen::VectorXd c = Eigen::Map<const Eigen::VectorXd>(s.c.data(), 3);
    const Eigen::VectorXd d = (Eigen::VectorXd(3) << 0.3, -1.0, 0.7).finished();
    const Reach r = LinearSys(system_matrix(s)).reach(set_of(s), 0.2, 1.0, 6);
    for (std::size_t k = 0; k < r.timePoint.size(); ++k) {
        const Eigen::MatrixXd flow = (A * 0.2 * double(k)).exp();
        const Eigen::VectorXd back = flow.transpose() * d;
        const double want = back.dot(c) + (back.transpose() * G).cwiseAbs().sum();
        check(close(support(r.timePoint[k], d), want, 1e-10), b + ": time point " + std::to_string(k));
    }
}

/// `x' = -x` on `[1, 3]`: the flow moves the set down, so the upper bound of a step is the
/// start's `3 e^{-t}`, tight to `dt²`. The lower bound is `e^{-t-dt}` at best; `linComb`
/// builds a symmetric zonotope and gives away up to `dt` on that far side.
void is_tight_in_one_dimension(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const double dt = s.timeStep;
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(system_matrix(s)).reach(set_of(s), dt, 1.0, 8, algorithm);
        for (std::size_t k = 0; k < r.timeInt.size(); ++k) {
            const double t0 = double(k) * dt;
            const double hi = support(r.timeInt[k], Eigen::VectorXd::Ones(1));
            const double lo = -support(r.timeInt[k], -Eigen::VectorXd::Ones(1));
            const std::string what = b + ": " + name(algorithm) + ": step " + std::to_string(k);
            check(hi >= 3 * std::exp(-t0) - 1e-12 && hi <= 3 * std::exp(-t0) + 1e-9 + 3 * dt * dt,
                  what + ": upper bound");
            check(lo <= std::exp(-t0 - dt) + 1e-12 && lo >= std::exp(-t0 - dt) - 3 * dt * std::exp(-t0),
                  what + ": lower bound");
        }
    }
}

/// Without dynamics nothing moves, so every enclosure is the initial set itself.
void a_zero_system_stays_put(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope X0 = Zonotope::generateRandom(2, 3, rng);
    const Eigen::VectorXd d = Eigen::VectorXd::Ones(2);
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(Tensor::zeros({2, 2})).reach(X0, 0.5, 2.0, 5, algorithm);
        for (const Zonotope &Z : r.timeInt)
            check(close(support(Z, d), support(X0, d), 1e-12), b + ": a zero system's enclosure grew");
    }
}

/// Each enclosure contains the time point at either end of its step.
void encloses_both_time_points(const std::string &b) {
    cora::Rng rng(7);
    const System &s = matlab_reference::three_dimensional();
    const Zonotope X0 = Zonotope::generateRandom(3, 4, rng);
    Eigen::MatrixXd dirs(3, 40);
    rng.normal(dirs.data(), dirs.size(), 1.0);
    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = LinearSys(system_matrix(s)).reach(X0, 0.2, 1.0, 6, algorithm);
        double worst = 1e9;
        for (std::size_t k = 0; k < r.timeInt.size(); ++k)
            for (int i = 0; i < 40; ++i) {
                const Eigen::VectorXd d = dirs.col(i);
                const double hull = support(r.timeInt[k], d);
                worst = std::min({worst, hull - support(r.timePoint[k], d),
                                  hull - support(r.timePoint[k + 1], d)});
            }
        check(worst >= -1e-9, b + ": " + name(algorithm) + ": a time point left its enclosure");
    }
}

/// The last step is a full one even when `tFinal` is not a multiple of the step.
void counts_steps_up(const std::string &b) {
    const System &s = matlab_reference::scalar();
    const LinearSys sys(system_matrix(s));
    check(sys.reach(set_of(s), 0.1, 0.25, 6).timeInt.size() == 3, b + ": ceil(0.25 / 0.1)");
    check(sys.reach(set_of(s), 0.1, 0.3, 6).timeInt.size() == 3, b + ": 0.3 / 0.1 stays 3");
    check(sys.reach(set_of(s), 0.1, 0.05, 6).timePoint.size() == 2, b + ": one step");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        matches_matlab_cora(b);
        encloses_the_trajectories(b);
        time_points_are_exact(b);
        is_tight_in_one_dimension(b);
        a_zero_system_stays_put(b);
        encloses_both_time_points(b);
        counts_steps_up(b);
    });
    return test::finish("linearSys reach");
}
