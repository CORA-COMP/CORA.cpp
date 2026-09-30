// What the tests share: a check, a tolerance, and a loop over the backends.
//
// A test written against `Tensor` runs once per backend with `for_each_backend`, so a
// property is checked on Eigen, on libtorch, and on the GPU when there is one — the same
// source, which is what "written once" promises.

#pragma once

#include "contSet/contSet.h"
#include "global/rng.h"
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
inline bool close(const cora::Tensor &a, const cora::Tensor &b, double tol = 1e-9) {
    if (a.shape() != b.shape()) return false;
    const std::vector<double> x = a.data(), y = b.data();
    for (std::size_t i = 0; i < x.size(); ++i)
        if (!close(x[i], y[i], tol)) return false;
    return true;
}

inline bool close(const cora::Tensor &a, const std::vector<double> &values, double tol = 1e-9) {
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
        cora::setBackend(name);
        body(name);
    }
    cora::setBackend("eigen");
}

/// The column (n, 1) of the given numbers, on the current backend.
inline cora::Tensor column(const std::vector<double> &v) {
    return cora::Tensor::fromData(v, {int64_t(v.size()), 1});
}

/// A random direction in n dimensions: standard normal entries.
inline std::vector<double> random_direction(cora::Rng &rng, int n) {
    std::vector<double> d(n);
    rng.normal(d.data(), d.size(), 1.0);
    return d;
}

/// The support function of any set along the host direction d, as a number.
inline double support(const cora::ContSet &S, const std::vector<double> &d) {
    return S.supportFunc(column(d)).data()[0];
}

/// Whether calling f throws.
template <class F>
bool throws(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

#ifdef CORACPP_TORCH
/// Runs `body(options, name)` on libtorch on the CPU and, when there is one, on the GPU;
/// `name` is "cpu" or "gpu" for the messages.
template <class F>
void for_each_device(F body) {
    std::vector<torch::TensorOptions> options{torch::TensorOptions().dtype(torch::kDouble)};
    std::vector<std::string> names{"cpu"};
    if (torch::cuda::is_available()) {
        options.push_back(torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, 0));
        names.push_back("gpu");
    }
    cora::setBackend("torch");
    torch::manual_seed(7);
    for (std::size_t i = 0; i < options.size(); ++i) body(options[i], names[i]);
    cora::setBackend("eigen");
}
#endif

inline int finish(const std::string &what) {
    if (failures == 0) std::cout << "all " << what << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

} // namespace test
