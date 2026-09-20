// The two ways CORA.cpp can run an instance.
//
// The Eigen backend is plain `double` on the CPU: no dispatch between an operation and
// its arithmetic, which is what makes the small instances fast, and templated on the
// scalar type so a differentiable one can be substituted at compile time. The libtorch
// backend runs the same operations on the CPU or a CUDA device and carries autograd with
// it, at the cost of a tensor dispatch per call.
//
// A backend only has to hand back a `Runner`: the instance's inputs, generated and bound,
// and the one call the harness measures.

#pragma once

#include <memory>
#include <string>

#include "catalog.h"
#include "mask.h"
#include "rng.h"

namespace cora {

struct Runner {
    virtual ~Runner() = default;

    /// One repetition of the measured operation; the containment answers when it has any.
    virtual Mask run() = 0;

    /// Blocks until the work `run` queued has actually finished, so a measurement covers
    /// work done. A no-op wherever the call was synchronous to begin with.
    virtual void sync() {}
};

/// Generates the instance's inputs and binds its operation, in `double` on the CPU.
std::unique_ptr<Runner> prepare_eigen(const Params &in, Rng &rng);

/// The libtorch backend. Every entry point is defined whether or not the tool was built
/// against libtorch; without it, `built()` is false and the rest declines.
namespace torch_backend {

/// Whether libtorch was compiled in at all.
bool built();

/// Whether the catalog's `device` can run here: `cpu` whenever built, `gpu` when the
/// worker also has a CUDA device.
bool supports(const std::string &device);

/// One line for `env`, naming the backend and what it found.
std::string describe();

/// Generates the instance's inputs on `in.device` and binds its operation.
std::unique_ptr<Runner> prepare(const Params &in);

/// Every operation once on each available device, and a gradient checked against the
/// analytic one. Throws if anything is wrong.
void warm_up();

/// Differentiates `supportFunc` on `device` and compares with the analytic gradient, so
/// that the differentiability the representations are for cannot rot unnoticed.
void check_gradients(const std::string &device);

} // namespace torch_backend

} // namespace cora
