// threads - the thread pool's width and the bar a parallel region must clear
//
// Syntax:   configure_threads();   int nt = threads_for(iterations, work);   set_num_threads(4);
//           EigenThreads narrow(work);
// Inputs:   iterations - the loop length; work - about how many scalar operations its body costs
// Outputs:  nt - the threads to use: the whole pool, or 1
// See also: threads.h

#include "cora/global/threads.h"
#include "global/tensor/derived.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <sstream>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Scalar operations a thread needs before it pays for being gathered into a region.
/// Since a region takes either one thread or the whole pool, the bar it has to clear is
/// this times the pool: below that, the barrier costs more than the work it divides.
constexpr long long kMinWorkPerThread = 1 << 12;

/// The same for a product Eigen splits itself, where the bar is far higher: gathering a
/// thread into a GEMM costs on the order of 100 us on a two-socket machine, so a thread
/// needs about a millisecond of arithmetic before it pays for itself.
constexpr long long kMinWorkPerEigenThread = 1 << 22;

std::atomic<int> configured{0};

/// The physical cores, counted from the kernel's topology: sibling hyperthreads share a
/// `thread_siblings_list`, so the distinct lists are the cores. Falls back to everything
/// the runtime reports when the topology is not readable.
int aux_physical_cores() {
    namespace fs = std::filesystem;
    std::set<std::string> cores;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator("/sys/devices/system/cpu", ec)) {
        const fs::path list = entry.path() / "topology" / "thread_siblings_list";
        std::ifstream file(list);
        std::string siblings;
        if (file && std::getline(file, siblings)) cores.insert(siblings);
    }
    if (!cores.empty()) return static_cast<int>(cores.size());
    const unsigned reported = std::thread::hardware_concurrency();
    return reported > 0 ? static_cast<int>(reported) : 1;
}

/// The whole text of a file, "" when it cannot be read.
std::string aux_read(const char *path) {
    std::ifstream file(path);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

/// The CPUs this process may run on; 0 where the system cannot say.
int aux_affinity() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) return CPU_COUNT(&set);
#endif
    return 0;
}

/// The default width: the cgroup quota (v2, else v1), the affinity mask and the physical cores.
int aux_default_threads() {
    int n = aux_physical_cores();
    const int quota = quota_cpus(aux_read("/sys/fs/cgroup/cpu.max"),
                                 aux_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
                                 aux_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"));
    if (quota > 0) n = std::min(n, quota);
    const int affinity = aux_affinity();
    if (affinity > 0) n = std::min(n, affinity);
    return std::max(1, n);
}

/// The threads asked for by the environment, 0 if neither variable holds a count.
/// CORACPP_NUM_THREADS comes before OMP_NUM_THREADS (whose first list entry counts).
int aux_requested() {
    for (const char *name : {"CORACPP_NUM_THREADS", "OMP_NUM_THREADS"}) {
        const char *value = std::getenv(name);
        const int n = value != nullptr ? std::atoi(value) : 0;
        if (n > 0) return n;
    }
    return 0;
}

/// Sets every pool to `n`.
void aux_apply(int n) {
#ifdef _OPENMP
    omp_set_num_threads(n);
#endif
    Eigen::setNbThreads(n);
    detail::set_backend_threads(n);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Thread count ------------------------------------------------------------------------------------

// The CPUs of a quota: ceil(quota / period), from whichever cgroup version has the text.
int quota_cpus(const std::string &cpu_max, const std::string &cfs_quota,
               const std::string &cfs_period) {
    double quota = 0, period = 0;
    if (!cpu_max.empty()) {
        std::istringstream(cpu_max) >> quota >> period; // "max <period>" fails to read: no quota
    } else {
        std::istringstream(cfs_quota) >> quota; // -1: no quota
        std::istringstream(cfs_period) >> period;
    }
    if (quota <= 0 || period <= 0) return 0;
    return std::max(1, static_cast<int>(std::ceil(quota / period)));
}

// Quota, affinity and cores: the least.
int available_threads() { return aux_default_threads(); }

// The pools, once: the environment's count, else the default.
void configure_threads() {
    if (configured != 0) return;
    const int requested = aux_requested();
    const int n = requested > 0 ? requested : aux_default_threads();
    configured = n;
    aux_apply(n);
}

// The pools to n, or to the configured default for n < 1.
void set_num_threads(int n) {
    const int requested = aux_requested();
    n = n >= 1 ? n : (requested > 0 ? requested : aux_default_threads());
    configured = n;
    aux_apply(n);
}

// The width the pools were set to.
int max_threads() {
    configure_threads();
    return configured;
}

// Regions -----------------------------------------------------------------------------------------

// A product gets as many threads as its work can keep busy, at most the pool.
EigenThreads::EigenThreads(long long work) : previous_(Eigen::nbThreads()) {
    const long long by_work = std::max<long long>(1, work / kMinWorkPerEigenThread);
    Eigen::setNbThreads(static_cast<int>(std::min<long long>(by_work, max_threads())));
}

// Back to the pool on the way out.
EigenThreads::~EigenThreads() { Eigen::setNbThreads(previous_); }

// The whole pool once the work clears the bar, else one thread: never a team in between.
int threads_for(long long iterations, long long work) {
    if (iterations <= 1 || work < kMinWorkPerThread) return 1; // no pool needed, none configured
    return work >= kMinWorkPerThread * max_threads() ? max_threads() : 1;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
