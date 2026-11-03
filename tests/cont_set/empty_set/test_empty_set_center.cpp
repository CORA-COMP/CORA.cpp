// test_empty_set_center - EmptySet center: the empty matrix (n, 0)
//
// CORA: contSet/@emptySet/center.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_center_has_no_columns(const std::string &b) {
    const Tensor c = EmptySet(3).center();
    check(c.shape() == std::vector<int64_t>({3, 0}), b + ": shape (3, 0)");
    check(EmptySet(0).center().shape() == std::vector<int64_t>({0, 0}), b + ": dimension 0");
}

/// center takes no input to refuse; with the checks off the result is the same. Also on lean.
void checks_and_their_switch() {
    const BackendScope lean("lean");
    check(EmptySet(2).center().shape() == std::vector<int64_t>({2, 0}), "center: on lean");
    ChecksScope off(false);
    check(EmptySet(2).center().shape() == std::vector<int64_t>({2, 0}),
          "center: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_center_has_no_columns);
    checks_and_their_switch();
    return test::finish("empty_set center");
}
