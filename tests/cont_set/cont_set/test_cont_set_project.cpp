// test_cont_set_project - ContSet::project: the dimensions are checked, the result is a Set

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// Dimensions outside the set are refused; the projection keeps the class.
void the_dimensions_are_checked() {
    const Zonotope Z =
        Zonotope(Tensor({1.0, 0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}, {0.0, 0.0}}));
    const ContSet &S = Z;
    check(test::raises([&] { S.project({0, 3}); }, Error::wrong_value), "dimension 3 of 3");
    check(test::raises([&] { S.project({}); }, Error::wrong_value), "no dimension");
    const Set P = S.project({1, 0});
    check(P.is<Zonotope>() && P.get().dim() == 2, "a zonotope of two dimensions");
    ChecksScope off(false);
    check(test::close(S.project({1, 0}).get().center(), P.get().center(), 0), "the checks off");
}

} // namespace

int main() {
    the_dimensions_are_checked();
    return test::finish("cont_set project");
}
