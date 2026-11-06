// threads - how wide a parallel region should be
//
// OpenMP and Eigen both default to every hyperthread the machine reports (160 on the
// competition's dual-socket worker, against 80 physical cores). The threads then oversubscribe
// their cores, and a region over a handful of sets spends milliseconds in the barrier that splits
// microseconds of work. Both are fixed here rather than at each call site: configure_threads sets
// the pool once, and threads_for skips a region whose work cannot pay for one.
//
// Syntax:   configure_threads();   int nt = threads_for(iterations, work);
//           EigenThreads narrow(work);   set_num_threads(4);
// See also: threads.cpp
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The CPUs a cgroup quota grants, rounded up, from the contents of the cgroup v2 file `cpu.max`
/// or the v1 files; 0 without quota. Pass empty strings for the version not in use.
int quota_cpus(const std::string &cpu_max, const std::string &cfs_quota,
               const std::string &cfs_period);

/// The threads CORA may use here: the least of the cgroup CPU quota, the CPU affinity mask and
/// the physical cores.
int available_threads();

/// Sets the width of every pool (OpenMP, Eigen, and libtorch's intra- and inter-op pools when
/// built with it) once, to the first of: `set_num_threads`, CORACPP_NUM_THREADS, OMP_NUM_THREADS,
/// `available_threads()`. Idempotent; called on first use.
void configure_threads();

/// Sets the width of every pool to `n`; `n < 1` goes back to the default above. (libtorch's
/// inter-op pool can be sized only once, so later calls leave it.)
void set_num_threads(int n);

/// The threads the pools were set to.
int max_threads();

/// How many threads a loop of `iterations` should use when its body costs about `work`
/// operations: the whole pool once there is enough work, otherwise one.
int threads_for(long long iterations, long long work);

/// Narrows Eigen's own threading to what a product of about `work` operations can use, restoring
/// the pool on the way out.
class EigenThreads {
  public:
    explicit EigenThreads(long long work);
    ~EigenThreads();

    EigenThreads(const EigenThreads &) = delete;
    EigenThreads &operator=(const EigenThreads &) = delete;

  private:
    int previous_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
