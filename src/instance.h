// One catalog instance: generate its inputs, then repeat the operation.
//
// `run_instance` is the whole instance, called by the daemon (or directly, when no daemon
// is up) inside the region the harness times. A batched instance holds its `batch_size`
// sets in one matrix, so every repetition is one call over the whole batch.

#pragma once

#include <iosfwd>
#include <string>

#include "catalog.h"

namespace cora {

/// Runs the instance described by `params` and writes its verdict; returns the verdict.
std::string run_instance(const std::string &params, const std::string &results_file,
                         std::ostream &log);

/// Runs every operation once on the Eigen backend, small: the first call into a code path
/// pays for the thread pool and the first allocations, a one-off cost of the process
/// rather than of any instance. Throws if anything is wrong.
void warm_up();

/// `warm_up`, plus the same for libtorch when it is compiled in: every operation on every
/// device it offers, and its gradients.
void warm_up_backends();

/// What the worker runs on.
void print_env(std::ostream &out);

void write_error(const std::string &results_file);

} // namespace cora
