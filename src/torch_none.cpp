// The libtorch backend when the tool was built without libtorch.
//
// Built in its place, so the rest of the tool never asks whether it exists: the Eigen
// backend still runs every `cpu` instance, and `gpu` instances report `unsupported`.

#include "backend.h"

#include <stdexcept>

namespace cora::torch_backend {

bool built() { return false; }

bool supports(const std::string &) { return false; }

std::string describe() { return "libtorch: not compiled in — gpu instances report unsupported"; }

std::unique_ptr<Runner> prepare(const Params &) {
    throw std::runtime_error("this build has no libtorch backend");
}

void warm_up() {}

void check_gradients(const std::string &) {}

} // namespace cora::torch_backend
