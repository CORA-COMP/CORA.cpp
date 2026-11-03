// test_fullspace_compact - Fullspace::compact_impl: the whole space is returned as it is
//
// CORA: contSet/@contSet/compact.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/set.h"
#include "testing.h"

using namespace cora;
using test::check;

int main() {
    const Fullspace S(3);
    // CORA returns before it reads the method and tol: any of them is accepted
    const Set R = static_cast<const ContSet &>(S).compact("anything", -1.0);
    check(R.is<Fullspace>() && R->dim() == S.dim(), "Fullspace: the same class and dimension");
    ChecksScope off(false);
    check(static_cast<const ContSet &>(S).compact().is<Fullspace>(), "the checks off");
    return test::finish("fullspace compact");
}
