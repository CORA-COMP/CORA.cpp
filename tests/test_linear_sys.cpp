// `linearSys` reachability on the Eigen backend: that both algorithms enclose the true
// trajectories, that an enclosure is tight where the answer is known, and that the two
// algorithms agree where they must.
//
// Backend-independent properties — batching, gradients, agreement between backends — are
// in test_linear_sys_torch.cpp.
//
//   make test

#include "contDynamics/linear_sys.h"
#include "contSet/sets.h"
#include "rng.h"
#include "tensor/eigen.h"

#include <unsupported/Eigen/MatrixFunctions>

#include <cmath>
#include <iostream>
#include <string>

using namespace cora::ct;
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

std::string name(Algorithm a) { return a == Algorithm::Standard ? "standard" : "wrapping-free"; }

/// A random zonotope with `m` generators in `n` dimensions, as CORA generates it.
Zonotope random_set(cora::Rng &rng, Index n, Index m) {
    const auto z = cora::random_zonotope(rng, n, m, 1);
    return {from_eigen(z.c), from_eigen(z.g)};
}

MatrixXd rotation_ish() {
    MatrixXd a(3, 3);
    a << -0.3, 1.0, 0.0, -1.0, -0.2, 0.4, 0.0, -0.5, -0.1;
    return a;
}

double support(const Zonotope &z, const VectorXd &d) {
    return to_eigen(z.support_func(from_eigen(d)))(0, 0);
}

/// One system and step size for the trajectory test.
struct Case {
    MatrixXd A;
    double dt;
    std::string what;
};

/// Points of the initial set pushed along the flow, at times inside every step, must lie
/// under the support function of the step's enclosure in every sampled direction. The fast
/// rotation turns a long way per step, so the flow bulges far out of the chord between the
/// time points and only the curvature enlargement keeps it inside.
void the_enclosures_contain_the_trajectories() {
    cora::Rng rng(3);
    const Index n = 3, m = 4;
    MatrixXd rotation = MatrixXd::Zero(n, n);
    rotation(0, 1) = 3.0;
    rotation(1, 0) = -3.0;
    rotation(2, 2) = -0.5;
    const Case cases[] = {{rotation_ish(), 0.1, "gentle"}, {rotation, 0.3, "fast rotation"}};

    for (const Case &tc : cases) {
        const Zonotope X0 = random_set(rng, n, m);
        const MatrixXd c0 = to_eigen(X0.c), g0 = to_eigen(X0.G);
        for (const Algorithm algorithm : kAlgorithms) {
            const Reach r = LinearSys(from_eigen(tc.A)).reach(X0, tc.dt, 2.0, 10, algorithm);
            const std::string what = tc.what + ": " + name(algorithm);
            check(r.time_point.size() == r.time_int.size() + 1, what + ": the step count");

            const Index samples = 400;
            MatrixXd beta(m, samples), dirs(n, samples), times(1, samples);
            rng.uniform(beta.data(), beta.size(), -1.0, 1.0);
            rng.normal(dirs.data(), dirs.size(), 1.0);
            rng.uniform(times.data(), times.size(), 0.0, 1.0);
            double worst = 1e9;
            for (std::size_t k = 0; k < r.time_int.size(); ++k) {
                for (Index i = 0; i < samples; ++i) {
                    const MatrixXd flow =
                        (tc.A * (static_cast<double>(k) + times(i)) * tc.dt).exp();
                    const VectorXd x = flow * (c0 + g0 * beta.col(i));
                    const VectorXd d = dirs.col(i);
                    worst = std::min(worst, (support(r.time_int[k], d) - d.dot(x)) / d.norm());
                }
            }
            check(worst >= -1e-9,
                  what + ": a trajectory left its enclosure by " + std::to_string(-worst));
        }
    }
}

/// `x' = -x` on `[1, 3]`: the flow moves the set down, so the upper bound of a step is the
/// start's `3 e^{-t}`, tight to `dt²`. The lower bound is `e^{-t-dt}` at best; `linComb`
/// builds a symmetric zonotope and gives away up to `dt` on that far side.
void the_enclosure_is_tight_in_one_dimension() {
    const Zonotope X0{from_eigen(MatrixXd::Constant(1, 1, 2.0)),
                      from_eigen(MatrixXd::Constant(1, 1, 1.0))};
    const double dt = 0.05;
    const VectorXd up = VectorXd::Ones(1), down = -up;

    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r =
            LinearSys(from_eigen(MatrixXd::Constant(1, 1, -1.0))).reach(X0, dt, 1.0, 8, algorithm);
        for (std::size_t k = 0; k < r.time_int.size(); ++k) {
            const double t0 = static_cast<double>(k) * dt;
            const double hi = support(r.time_int[k], up), lo = -support(r.time_int[k], down);
            check(hi >= 3 * std::exp(-t0) - 1e-12 && hi <= 3 * std::exp(-t0) + 1e-9 + 3 * dt * dt,
                  name(algorithm) + ": upper bound of step " + std::to_string(k));
            check(lo <= std::exp(-t0 - dt) + 1e-12 &&
                      lo >= std::exp(-t0 - dt) - 3 * dt * std::exp(-t0),
                  name(algorithm) + ": lower bound of step " + std::to_string(k));
        }
    }
}

/// Without dynamics nothing moves, so every enclosure is the initial set itself.
void a_zero_system_stays_put() {
    cora::Rng rng(5);
    const Zonotope X0 = random_set(rng, 2, 3);
    const Reach r = LinearSys(from_eigen(MatrixXd::Zero(2, 2))).reach(X0, 0.5, 2.0, 5);
    const VectorXd d = VectorXd::Ones(2);
    for (const Zonotope &set : r.time_int)
        check(std::abs(support(set, d) - support(X0, d)) < 1e-12, "a zero system's enclosure grew");
}

/// The time-point sets are the same set either way; only the enclosures between differ.
void the_algorithms_share_their_time_points() {
    cora::Rng rng(7);
    const Zonotope X0 = random_set(rng, 3, 4);
    const LinearSys sys(from_eigen(rotation_ish()));
    const Reach s = sys.reach(X0, 0.1, 1.0, 10, Algorithm::Standard);
    const Reach w = sys.reach(X0, 0.1, 1.0, 10, Algorithm::WrappingFree);
    for (std::size_t k = 0; k < s.time_point.size(); ++k) {
        check(
            (to_eigen(s.time_point[k].c) - to_eigen(w.time_point[k].c)).cwiseAbs().maxCoeff() <
                    1e-9 &&
                (to_eigen(s.time_point[k].G) - to_eigen(w.time_point[k].G)).cwiseAbs().maxCoeff() <
                    1e-9,
            "the time-point sets differ at " + std::to_string(k));
    }
}

/// Values computed by MATLAB CORA (R2024b, v2026.1.0) for `linearSys([-0.2 1; -1 -0.2])`, the
/// initial set `zonotope([1; 0.5], [0.1 0; 0 0.2])`, `timeStep = 0.1`, `taylorTerms = 8`:
/// the correction matrix `F` and the support of every reachable set along four directions.
/// Without inputs CORA's two algorithms give the same sets, and they are the wrapping-free
/// ones here; the standard algorithm agrees on the first step, where `F X0` is the same.
void the_results_match_matlab_cora() {
    const double f_inf[2][2] = {{-3.9476137617237179e-05, -1.5390117296157174e-06},
                                {-0.0005564528717124116, -3.9476137617237179e-05}};
    const double f_sup[2][2] = {{0.0012000413732381331, 0.0005564528717124116},
                                {1.5390117296157174e-06, 0.0012000413732381331}};
    const double time_int[5][4] = {
        {1.14551093304521, 0.710682871515058, 1.81718295345275, 0.765085217909499},
        {1.17453307773407, 0.60483335615097, 1.73067562161623, 0.983574784598129},
        {1.1904535553181, 0.49697347929813, 1.62993177830263, 1.18347864499177},
        {1.20249549452805, 0.3882807197523, 1.51653581320245, 1.36348935159334},
        {1.20352688786842, 0.279894870376489, 1.39214903709877, 1.52255335260261}};
    const double time_point[6][4] = {
        {1.1, 0.7, 1.8, 0.5},
        {1.14133154679847, 0.59464030962414, 1.71640053991283, 0.732292337742362},
        {1.16941695585957, 0.487354876938997, 1.6185959538163, 0.948017296106397},
        {1.18448954012092, 0.379312011536149, 1.50813946173885, 1.14562713552654},
        {1.18690601169904, 0.271641928077119, 1.38665225230257, 1.32381932321135},
        {1.17713800989948, 0.165426727747973, 1.25580430434923, 1.48154018593495}};
    const VectorXd dirs[4] = {(VectorXd(2) << 1, 0).finished(), (VectorXd(2) << 0, 1).finished(),
                              (VectorXd(2) << 1, 1).finished(), (VectorXd(2) << 1, -2).finished()};

    MatrixXd a(2, 2);
    a << -0.2, 1.0, -1.0, -0.2;
    MatrixXd g(2, 2);
    g << 0.1, 0.0, 0.0, 0.2;
    const Zonotope X0{from_eigen((VectorXd(2) << 1.0, 0.5).finished()), from_eigen(g)};
    const LinearSys sys(from_eigen(a));

    const IntervalMatrix F = sys.correction_matrix_state(0.1, 8);
    const MatrixXd centre = to_eigen(F.center), radius = to_eigen(F.radius);
    double f_err = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            f_err = std::max(f_err, std::abs(centre(i, j) - radius(i, j) - f_inf[i][j]));
            f_err = std::max(f_err, std::abs(centre(i, j) + radius(i, j) - f_sup[i][j]));
        }
    check(f_err < 1e-14, "F differs from MATLAB CORA by " + std::to_string(f_err));

    for (const Algorithm algorithm : kAlgorithms) {
        const Reach r = sys.reach(X0, 0.1, 0.5, 8, algorithm);
        double ti_err = 0.0, tp_err = 0.0;
        for (int k = 0; k < 5; ++k)
            for (int i = 0; i < 4; ++i) {
                // Only the first step of the standard algorithm is CORA's.
                if (algorithm == Algorithm::WrappingFree || k == 0)
                    ti_err = std::max(ti_err,
                                      std::abs(support(r.time_int[k], dirs[i]) - time_int[k][i]));
            }
        for (int k = 0; k < 6; ++k)
            for (int i = 0; i < 4; ++i)
                tp_err = std::max(tp_err,
                                  std::abs(support(r.time_point[k], dirs[i]) - time_point[k][i]));
        check(ti_err < 1e-12, name(algorithm) + ": the enclosures differ from MATLAB CORA by " +
                                  std::to_string(ti_err));
        check(tp_err < 1e-12, name(algorithm) +
                                  ": the time-point sets differ from MATLAB CORA by " +
                                  std::to_string(tp_err));
    }
}

} // namespace

int main() {
    the_enclosures_contain_the_trajectories();
    the_enclosure_is_tight_in_one_dimension();
    a_zero_system_stays_put();
    the_algorithms_share_their_time_points();
    the_results_match_matlab_cora();
    if (failures == 0) std::cout << "all linearSys tests passed\n";
    return failures == 0 ? 0 : 1;
}
