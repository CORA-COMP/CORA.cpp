// test_fullspace_complement - fullspace complement: the empty set
//
// CORA: contSet/@fullspace/not.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_complement_is_the_empty_set(const std::string &b) {
    check(Fullspace(3).complement().dim() == 3 && Fullspace(0).complement().dim() == 0,
          b + ": the dimension is kept");
    check(Fullspace(2).complement().isequal(EmptySet(2)), b + ": the empty set");
    check(Fullspace(2).complement().complement().isequal(Fullspace(2)), b + ": twice gives it back");
}

/// complement takes no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(Fullspace(2).complement().dim() == 2, "complement: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_complement_is_the_empty_set);
    checks_and_their_switch();
    return test::finish("fullspace complement");
}
