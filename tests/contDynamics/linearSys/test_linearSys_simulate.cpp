// linearSys.simulate on every backend: exact against the matrix exponential, and inside the
// reachable sets — the sanity check every reachability tool is measured against.

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/linearSys/matlabReference.h"
#include "global/rng.h"
#include "testing.h"

#include <unsupported/Eigen/MatrixFunctions>

using namespace cora::ct;
using test::check;
using test::close;

namespace {

Tensor tensor_of(const Eigen::MatrixXd &M) {
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> rows = M;
    return Tensor::from_data({rows.data(), rows.data() + rows.size()}, {M.rows(), M.cols()});
}

Eigen::MatrixXd host_of(const Tensor &t) {
    const std::vector<int64_t> s = t.shape();
    const std::vector<double> v = t.data();
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(v.data(), s[0], s[1]);
}

Eigen::MatrixXd oscillator3() {
    Eigen::MatrixXd A(3, 3);
    A << -0.3, 1.0, 0.0, -1.0, -0.2, 0.4, 0.0, -0.5, -0.1;
    return A;
}

void follows_the_matrix_exponential(const std::string &b) {
    const Eigen::MatrixXd A = oscillator3();
    Eigen::MatrixXd x0(3, 4);
    x0 << 1, 0, -1, 0.5, 0, 1, 0.2, -0.5, 0.3, -0.2, 1, 0;
    const double dt = 0.15;
    const std::vector<Tensor> x = LinearSys(tensor_of(A)).simulate(tensor_of(x0), dt, 1.0);

    check(x.size() == 8, b + ": ceil(1.0 / 0.15) + 1 time points");
    check(close(x[0], tensor_of(x0)), b + ": the start is the start");
    double worst = 0.0;
    for (std::size_t k = 0; k < x.size(); ++k)
        worst = std::max(worst, (host_of(x[k]) - (A * dt * double(k)).exp() * x0).cwiseAbs().maxCoeff());
    check(worst < 1e-12, b + ": a trajectory differs from e^{A t} x0 by " + std::to_string(worst));
}

/// A stable linear system contracts: the norm of every trajectory falls.
void a_stable_system_decays(const std::string &b) {
    const Tensor A({{-1.0, 0.0}, {0.0, -2.0}});
    const std::vector<Tensor> x = LinearSys(A).simulate(Tensor({{1.0}, {1.0}}), 0.1, 3.0);
    const Eigen::MatrixXd last = host_of(x.back());
    check(std::abs(last(0, 0) - std::exp(-3.0)) < 1e-12 && std::abs(last(1, 0) - std::exp(-6.0)) < 1e-12,
          b + ": x' = -x and x' = -2x decay exactly");
}

double support(const Zonotope &Z, const Eigen::VectorXd &d) {
    return Z.support_func(Tensor::from_data({d.data(), d.data() + d.size()}, {d.size(), 1})).data()[0];
}

/// Simulated points lie in the reachable sets: at every time point in the time-point set, and
/// at every point in between in the enclosure of that step. Starting from extreme points of the
/// initial set is the demanding case: the boundary is where a set that is too small shows it.
void simulations_stay_in_the_reachable_set(const std::string &b) {
    cora::Rng rng(11);
    const int n = 3, N = 60, refine = 5;
    const double dt = 0.2, t_final = 2.0;
    const Tensor A = tensor_of(oscillator3());
    const Zonotope X0 = Zonotope::generate_random(n, 4, rng);
    const LinearSys sys(A);

    Eigen::MatrixXd dirs(n, 40);
    rng.normal(dirs.data(), dirs.size(), 1.0);

    for (const Algorithm algorithm : {Algorithm::Standard, Algorithm::WrappingFree}) {
        const Reach R = sys.reach(X0, dt, t_final, 8, algorithm);
        for (const bool extreme : {false, true}) {
            const Tensor start = X0.rand_point(N, rng, extreme);
            const std::vector<Tensor> x = sys.simulate(start, dt / refine, t_final);
            double worst = 1e9;
            for (std::size_t j = 0; j < x.size(); ++j) {
                const Eigen::MatrixXd points = host_of(x[j]);
                const std::size_t step = std::min(j / refine, R.time_int.size() - 1);
                for (int i = 0; i < 40; ++i) {
                    const Eigen::VectorXd d = dirs.col(i);
                    const double bound = support(R.time_int[step], d);
                    worst = std::min(worst, (bound - (d.transpose() * points).maxCoeff()) / d.norm());
                    if (j % refine == 0) {
                        const double at = support(R.time_point[j / refine], d);
                        worst = std::min(worst, (at - (d.transpose() * points).maxCoeff()) / d.norm());
                    }
                }
            }
            check(worst >= -1e-9, b + ": " + (algorithm == Algorithm::Standard ? "standard" : "wrapping-free") +
                                      (extreme ? ", extreme" : ", random") + ": a trajectory left the reachable set by " +
                                      std::to_string(-worst));
        }
    }
}

/// `simulate_random` draws from the initial set: same seed, same trajectories.
void simulate_random_is_seeded(const std::string &b) {
    const Zonotope X0(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.1, 0.1}}));
    const LinearSys sys(tensor_of(oscillator3()));
    cora::Rng first(5), second(5), other(6);
    const std::vector<Tensor> a = sys.simulate_random(X0, 8, 0.1, 1.0, first);
    const std::vector<Tensor> c = sys.simulate_random(X0, 8, 0.1, 1.0, second);
    const std::vector<Tensor> d = sys.simulate_random(X0, 8, 0.1, 1.0, other);
    check(a.size() == 11 && a[0].shape() == std::vector<int64_t>({3, 8}), b + ": shapes");
    check(close(a.back(), c.back()), b + ": the same seed gives the same trajectories");
    check(!close(a.back(), d.back()), b + ": another seed gives others");
    // The initial set is an interval too: any ContSet works.
    const Interval box(Tensor({0.9, -0.1, -0.1}), Tensor({1.1, 0.1, 0.1}));
    cora::Rng rng(1);
    check(sys.simulate_random(box, 5, 0.1, 0.5, rng).size() == 6, b + ": simulate_random from an interval");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        follows_the_matrix_exponential(b);
        a_stable_system_decays(b);
        simulations_stay_in_the_reachable_set(b);
        simulate_random_is_seeded(b);
    });
    return test::finish("linearSys simulate");
}
