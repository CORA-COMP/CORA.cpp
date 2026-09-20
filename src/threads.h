// How wide a parallel region should be.
//
// OpenMP and Eigen both default to every hyperthread the machine reports — 160 on the
// competition's dual-socket worker, against 80 physical cores. Two things go wrong with
// that: the threads oversubscribe their cores, and a region over a handful of sets
// spends milliseconds in the barrier that splits microseconds of work. Both are fixed
// here rather than at each call site: `configure_threads` sets the pool once, and
// `threads_for` skips a region whose work cannot pay for one.

#pragma once

namespace cora {

/// Sets OpenMP and Eigen to the machine's physical cores. An explicit `OMP_NUM_THREADS`
/// is left alone — an operator who set it meant it. Idempotent; called at startup.
void configure_threads();

/// The threads the pool was set to.
int max_threads();

/// How many threads a loop of `iterations` should use when its whole body costs about
/// `work` scalar operations: the whole pool once there is enough work to keep all of it
/// busy, otherwise one.
///
/// Deliberately all-or-nothing. A team narrower than the pool is not cheaper — OpenMP
/// parks and wakes the difference, and between two regions of one repetition that costs
/// more than either region: a batch of ten sets whose draw fans out to the pool and whose
/// product asks for ten threads spends 1.25 s where one width spends 0.011 s. The price
/// is that the pool is the only alternative to serial, so the bar to clear rises with it.
int threads_for(long long iterations, long long work);

/// Narrows Eigen's own threading to what a product of about `work` scalar operations can
/// use, restoring the pool on the way out.
///
/// Eigen splits a product on its own threshold, which is far too eager for a wide
/// machine: a 40 MFLOP product divided 80 ways across two sockets spends an order of
/// magnitude longer gathering the threads than doing the arithmetic. Capping by the work
/// is the difference between 33 s and 0.1 s on the catalog's mid-sized `matMul`.
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
