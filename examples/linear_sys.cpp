// Reachability of a damped oscillator with both algorithms. Nothing here names a library:
// the backend is set once — below, or from outside with CORACPP_BACKEND=eigen|torch|torch:cuda
// — and every tensor is made on it.
//
//   make TORCH=... example     then     CORACPP_BACKEND=eigen build/example_linear_sys

#include "contDynamics/linear_sys.h"

#include <iostream>

using namespace cora::ct;

int main() {
    // set_backend("eigen");

    const Tensor A({{-0.1, 1.0}, {-1.0, -0.1}});
    const Zonotope X0{Tensor({1.0, 0.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}})};
    const Tensor d({1.0, 0.0}); // the direction whose extent is reported

    std::cout << "backend: " << backend().name() << "\n";
    for (const Algorithm algorithm : {Algorithm::Standard, Algorithm::WrappingFree}) {
        const Reach R = LinearSys(A).reach(X0, 0.1, 1.0, 10, algorithm);
        std::cout << (algorithm == Algorithm::Standard ? "standard" : "wrapping-free")
                  << ": max x1 per step\n";
        for (std::size_t k = 0; k < R.time_int.size(); ++k)
            std::cout << "  [" << 0.1 * k << ", " << 0.1 * (k + 1) << "]  "
                      << R.time_int[k].support_func(d).data()[0] << "\n";
    }
}
