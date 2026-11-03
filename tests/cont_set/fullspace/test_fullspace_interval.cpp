// test_fullspace_interval - fullspace interval: the box (-Inf, Inf)^n
//
// CORA: contSet/@fullspace/interval.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void interval_has_infinite_bounds(const std::string &b) {
    const Interval I = Fullspace(2).interval();
    check(I.inf().data() == std::vector<double>({-kInf, -kInf}) &&
              I.sup().data() == std::vector<double>({kInf, kInf}),
          b + ": (-Inf, Inf)^2");
    check(Fullspace(2).vertices()[0].size() == 4, b + ": four corners in the plane");
}

/// R^0 has no box, as CORA; the checks off give the same answer elsewhere.
void checks_and_their_switch() {
    check(test::raises([] { Fullspace(0).interval(); }, Error::out_of_domain), "interval: R^0");
    ChecksScope off(false);
    check(Fullspace(1).interval().dim() == 1, "interval: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(interval_has_infinite_bounds);
    checks_and_their_switch();
    return test::finish("fullspace interval");
}
