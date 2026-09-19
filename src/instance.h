// One catalog instance: generate its inputs, then repeat the operation.
//
// `run_instance` is the whole instance, called by the daemon (or directly, when no daemon
// is up) inside the region the harness times. A batched instance holds its `batch_size`
// sets in one matrix, so every repetition is one call over the whole batch.

#pragma once

#include <iosfwd>
#include <string>

namespace cora {

inline constexpr const char *kOperations[] = {
    "startup", "generateRandom", "randPoint", "supportFunc", "matMul", "minkSum", "contains"};

/// Seed of every instance, so a warm daemon behaves like a fresh process.
inline constexpr unsigned long long kSeed = 0;

/// Runs the instance described by `params` and writes its verdict; returns the verdict.
std::string run_instance(const std::string &params, const std::string &results_file,
                         std::ostream &log);

/// Runs every operation once, small: the first call into a code path pays for the thread
/// pool and the first allocations, a one-off cost of the process rather than of any
/// instance. Throws if anything is wrong.
void warm_up();

/// What the worker runs on.
void print_env(std::ostream &out);

void write_error(const std::string &results_file);

} // namespace cora
