// test_empty_set_vertices_3d - EmptySet vertices_3d: none
//
// CORA: contSet/@emptySet/vertices_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_empty_set_has_no_corner(const std::string &b) {
    check(EmptySet(3).vertices_3d().empty(), b + ": no list of corners");
}

/// A set of two dimensions has no corners in space; with the checks off the result is the same.
void checks_and_their_switch() {
    check(test::raises([] { EmptySet(2).vertices_3d(); }, Error::not_supported),
          "vertices_3d: two dimensions");
    ChecksScope off(false);
    check(EmptySet(3).vertices_3d().empty(), "vertices_3d: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_empty_set_has_no_corner);
    checks_and_their_switch();
    return test::finish("empty_set vertices_3d");
}
