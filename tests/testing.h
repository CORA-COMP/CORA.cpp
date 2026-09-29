// What the tests share: a check, a tolerance, and a loop over the backends.
//
// A test written against `ct::Tensor` runs once per backend with `for_each_backend`, so a
// property is checked on Eigen, on libtorch, and on the GPU when there is one — the same
// source, which is what "written once" promises.

#pragma once

#include "tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#ifdef CORACPP_TORCH
#include <torch/torch.h>
#endif

namespace test {

inline int failures = 0;

inline void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

inline bool close(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) <= tol * (1.0 + std::max(std::abs(a), std::abs(b)));
}

/// Whether two tensors have the same shape and values up to `tol`.
inline bool close(const cora::ct::Tensor &a, const cora::ct::Tensor &b, double tol = 1e-9) {
    if (a.shape() != b.shape()) return false;
    const std::vector<double> x = a.data(), y = b.data();
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!close(x[i], y[i], tol)) return false;
    return true;
}

inline bool close(const cora::ct::Tensor &a, const std::vector<double> &values, double tol = 1e-9) {
    const std::vector<double> x = a.data();
    if (x.size() != values.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!close(x[i], values[i], tol)) return false;
    return true;
}

/// Every backend this build and this machine offer, as `setBackend` names them.
inline std::vector<std::string> backends() {
    std::vector<std::string> all{"eigen"};
#ifdef CORACPP_TORCH
    all.push_back("torch");
    if (torch::cuda::is_available()) all.push_back("torch:cuda");
#endif
    return all;
}

/// Runs `body(name)` with each backend set in turn; `check` messages carry the name.
template <class F>
void for_each_backend(F body) {
    for (const std::string &name : backends()) {
        cora::ct::setBackend(name);
        body(name);
    }
    cora::ct::setBackend("eigen");
}

inline int finish(const std::string &what) {
    if (failures == 0) std::cout << "all " << what << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

} // namespace test
