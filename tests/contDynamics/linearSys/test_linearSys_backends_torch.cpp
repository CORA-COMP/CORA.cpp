// The same code on different backends and devices gives the same answer: the algorithms are
// written once, so this is what tells the backends apart.

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "testing.h"

#include <cmath>

using namespace cora::ct;
using test::check;

namespace {

/// Everything a run produces, flattened, from code that names no backend or device.
std::vector<double> run(Algorithm algorithm) {
    const Tensor A({{-0.2, 1.0, 0.0}, {-1.0, -0.2, 0.3}, {0.0, -0.4, -0.1}});
    const Zonotope X0(Tensor({1.0, 0.5, -0.3}),
                      Tensor({{0.1, 0.0, 0.05}, {0.0, 0.2, 0.0}, {0.1, 0.0, 0.1}}));
    const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 8, algorithm);
    std::vector<double> all;
    for (const auto *sets : {&R.timeInt, &R.timePoint})
        for (const Zonotope &Z : *sets)
            for (const Tensor *t : {&Z.c, &Z.G}) {
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
    for (const Algorithm algorithm : {Algorithm::Standard, Algorithm::WrappingFree}) {
        const std::string name = algorithm == Algorithm::Standard ? "standard" : "wrapping-free";
        cora::ct::setBackend("eigen");
        const std::vector<double> eigen = run(algorithm);
        for (const std::string &other : test::backends()) {
            if (other == "eigen") continue;
            cora::ct::setBackend(other);
            check(max_diff(eigen, run(algorithm)) < 1e-10, name + ": eigen and " + other + " differ");
        }
    }

    // A specification is answered alike everywhere.
    std::vector<bool> answers;
    test::for_each_backend([&](const std::string &) {
        const Zonotope X0(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
        const Reach R = LinearSys(Tensor({{-0.1, 1.0}, {-1.0, -0.1}})).reach(X0, 0.1, 2.0, 8);
        answers.push_back(Specification::safeSet(Tensor({1.0, 0.0}), 1.55).check(R.timeInt));
        answers.push_back(Specification::unsafeSet(Tensor({0.0, 1.0}), -1.4).check(R.timeInt));
    });
    for (std::size_t i = 2; i < answers.size(); ++i)
        check(answers[i] == answers[i % 2], "a specification is answered differently on another backend");

    cora::ct::setBackend("eigen");
    return test::finish("linearSys backends");
}
