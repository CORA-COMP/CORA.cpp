// testing - what the tests share: a check, a tolerance, a loop over the backends, the errors
//
// A test written against `Tensor` runs once per backend with `for_each_backend`, so a
// property is checked on Eigen, on libtorch, and on the GPU when there is one — the same
// source, which is what "written once" promises. `raises(f, Error::dimension_mismatch)` checks
// that a wrong input is refused with the right identifier; a `cora::ChecksScope off(false)`
// block runs a valid call without the checks.

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/error.h"
#include "cora/global/settings.h"
#include "cora/global/rng.h"
#include "cora/global/tensor/tensor.h"

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

/// Every backend this build and this machine offer, as `BackendScope` names them.
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
        const cora::BackendScope scope(name);
        body(name);
    }
}

/// The column (n, 1) of the given numbers, on the current backend.
inline cora::Tensor column(const std::vector<double> &v) {
    return cora::Tensor::from_data(v, {int64_t(v.size()), 1});
}

/// A random direction in n dimensions: standard normal entries.
inline std::vector<double> random_direction(cora::Rng &rng, int n) {
    std::vector<double> d(n);
    rng.normal(d.data(), d.size(), 1.0);
    return d;
}

/// The support function of any set along the host direction d, as a number.
inline double support(const cora::ContSet &S, const std::vector<double> &d) {
    return S.support_function(column(d)).data()[0];
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

/// Whether calling f raises a cora::Error with the identifier `id`.
template <class F>
bool raises(F f, cora::Error::Id id) {
    try {
        f();
    } catch (const cora::Error &e) {
        if (e.id() != id)
            std::cerr << "  raised " << e.identifier() << " instead of "
                      << cora::Error::identifier(id) << ": " << e.what() << "\n";
        return e.id() == id;
    } catch (const std::exception &e) {
        std::cerr << "  raised a non-CORA error: " << e.what() << "\n";
        return false;
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
    const cora::BackendScope scope("torch");
    torch::manual_seed(7);
    for (std::size_t i = 0; i < options.size(); ++i) body(options[i], names[i]);
}
#endif

inline int finish(const std::string &what) {
    if (failures == 0) std::cout << "all " << what << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

} // namespace test
