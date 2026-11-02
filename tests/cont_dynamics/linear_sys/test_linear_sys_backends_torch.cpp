// test_linear_sys_backends_torch - the same code on different backends and devices gives the same
// answer: the algorithms are written once, so this is what tells the backends apart.

#include "cont_dynamics/linear_sys/linear_sys_testing.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/specification/specification.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

namespace {

/// Everything a run produces, flattened, from code that names no backend or device.
std::vector<double> run(const std::string &algorithm) {
    const Tensor A({{-0.2, 1.0, 0.0}, {-1.0, -0.2, 0.3}, {0.0, -0.4, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.5, -0.3}),
                      Tensor({{0.1, 0.0, 0.05}, {0.0, 0.2, 0.0}, {0.1, 0.0, 0.1}}));
    const ReachSet R = LinearSys(A).reach({.initial_set = X0, .t_final = 1.0},
                                          test::lin::algorithm(algorithm, 0.1, 8));
    std::vector<double> all;
    for (const auto *sets : {&R.time_interval, &R.time_point})
        for (const Zonotope &Z : *sets)
            for (const Tensor *t : {&Z.center(), &Z.generators()}) {
                const std::vector<double> v = t->data();
                all.insert(all.end(), v.begin(), v.end());
            }
    return all;
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) return 1e300;
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

} // namespace

int main() {
    for (const std::string &algorithm : test::lin::kAlgorithms) {
        const std::string &name = algorithm;
        std::vector<double> eigen;
        {
            const BackendScope scope("eigen");
            eigen = run(algorithm);
        }
        for (const std::string &other : test::backends()) {
            if (other == "eigen") continue;
            const BackendScope scope(other);
            check(max_diff(eigen, run(algorithm)) < 1e-10,
                  name + ": eigen and " + other + " differ");
        }
    }

    // A specification is answered alike everywhere.
    std::vector<bool> answers;
    test::for_each_backend([&](const std::string &) {
        const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
        const ReachSet R = LinearSys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}}))
                               .reach({.initial_set = X0, .t_final = 2.0},
                                      alg::Standard{.time_step = 0.1, .taylor_terms = 8});
        answers.push_back(Specification::safe_set(Tensor({1.0, 0.0}), 1.55).check(R.time_interval));
        answers.push_back(
            Specification::unsafe_set(Tensor({0.0, 1.0}), -1.4).check(R.time_interval));
    });
    for (std::size_t i = 2; i < answers.size(); ++i)
        check(answers[i] == answers[i % 2],
              "a specification is answered differently on another backend");

    return test::finish("linear_sys backends");
}
