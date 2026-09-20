#include "threads.h"

#include <Eigen/Core>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cora {
namespace {

/// Below this many scalar operations per thread a region is not worth splitting: the
/// barrier across a wide pool costs more than the work it divides.
constexpr long long kMinWorkPerThread = 1 << 12;

/// The same for a product Eigen splits itself, where the bar is far higher: gathering a
/// thread into a GEMM costs on the order of 100 us on a two-socket machine, so a thread
/// needs about a millisecond of arithmetic before it pays for itself.
constexpr long long kMinWorkPerEigenThread = 1 << 22;

int configured = 0;

/// The physical cores, counted from the kernel's topology: sibling hyperthreads share a
/// `thread_siblings_list`, so the distinct lists are the cores. Falls back to everything
/// the runtime reports when the topology is not readable.
int physical_cores() {
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

} // namespace

void configure_threads() {
    if (configured != 0) return;
    const char *asked = std::getenv("OMP_NUM_THREADS");
#ifdef _OPENMP
    configured = asked != nullptr ? omp_get_max_threads() : physical_cores();
    if (asked == nullptr) omp_set_num_threads(configured);
#else
    configured = asked != nullptr ? std::atoi(asked) : physical_cores();
#endif
    configured = std::max(1, configured);
    Eigen::setNbThreads(configured);
}

int max_threads() {
    configure_threads();
    return configured;
}

EigenThreads::EigenThreads(long long work) : previous_(Eigen::nbThreads()) {
    const long long by_work = std::max<long long>(1, work / kMinWorkPerEigenThread);
    Eigen::setNbThreads(static_cast<int>(std::min<long long>(by_work, max_threads())));
}

EigenThreads::~EigenThreads() { Eigen::setNbThreads(previous_); }

int threads_for(long long iterations, long long work) {
    if (iterations <= 1) return 1;
    const long long by_work = std::max<long long>(1, work / kMinWorkPerThread);
    const long long limit = std::min<long long>(iterations, by_work);
    return static_cast<int>(std::min<long long>(limit, max_threads()));
}

} // namespace cora
