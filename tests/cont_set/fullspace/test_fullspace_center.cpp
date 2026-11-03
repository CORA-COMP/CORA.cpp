// test_fullspace_center - fullspace center: the origin
//
// CORA: contSet/@fullspace/center.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void center_is_the_origin(const std::string &b) {
    const Tensor c = Fullspace(3).center();
    check(c.shape() == std::vector<int64_t>({3, 1}) && c.data() == std::vector<double>(3, 0.0),
          b + ": the origin of R^3 as a column");
    check(Fullspace(0).center().shape() == std::vector<int64_t>({0, 1}),
          b + ": the empty column in R^0 (CORA: NaN)");
}

/// center has no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(Fullspace(2).center().numel() == 2, "center: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(center_is_the_origin);
    checks_and_their_switch();
    return test::finish("fullspace center");
}
