// test_threads - the thread count: the cgroup quota parser, the default, and set_num_threads
//
// CORA.cpp only: the policy of the repository

#include "cora/global/threads.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void quota_parser() {
    check(quota_cpus("1600000 100000\n", "", "") == 16, "v2: quota over period");
    check(quota_cpus("150000 100000", "", "") == 2, "v2: a fraction rounds up");
    check(quota_cpus("50000 100000", "", "") == 1, "v2: at least one CPU");
    check(quota_cpus("max 100000\n", "", "") == 0, "v2: max is no quota");
    check(quota_cpus("max", "", "") == 0, "v2: no period");
    check(quota_cpus("", "1600000\n", "100000\n") == 16, "v1: quota over period");
    check(quota_cpus("", "-1\n", "100000\n") == 0, "v1: -1 is no quota");
    check(quota_cpus("", "200000", "") == 0, "v1: no period");
    check(quota_cpus("", "", "") == 0, "nothing readable is no quota");
    check(quota_cpus("garbage 100000", "", "") == 0, "unreadable is no quota");
}

void setting() {
    check(available_threads() >= 1, "at least one thread is available");
    set_num_threads(3);
    check(max_threads() == 3, "set_num_threads sets the pools");
    check(threads_for(1000, 1LL << 40) == 3, "... and the width of a region");
    set_num_threads(0);
    check(max_threads() >= 1, "0 goes back to the default");
}

} // namespace

int main() {
    quota_parser();
    setting();
    return test::finish("threads");
}
