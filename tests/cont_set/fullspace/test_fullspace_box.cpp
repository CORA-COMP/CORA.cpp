// test_fullspace_box - fullspace box: the whole space is its own box
//
// CORA: contSet/@fullspace/box.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void box_is_the_set_itself() {
    check(Fullspace(3).box().dim() == 3 && Fullspace(0).box().dim() == 0, "box: the same dimension");
    check(Fullspace(3).box().isequal(Fullspace(3)), "box: still the whole space");
}

/// box has no input to refuse; with the checks off the answer is the same.
void checks_and_their_switch() {
    ChecksScope off(false);
    check(Fullspace(2).box().dim() == 2, "box: the same with the checks off");
}

} // namespace

int main() {
    box_is_the_set_itself();
    checks_and_their_switch();
    return test::finish("fullspace box");
}
