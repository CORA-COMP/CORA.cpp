// test_zonotope_origin - zonotope origin: the zonotope {0}
//
// CORA: contSet/@zonotope/origin.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The center is 0 (n, 1) and there is no generator; the set contains only the origin.
void origin_is_the_point_zero(const std::string &b) {
    for (const int n : {1, 2, 5}) {
        const Zonotope Z = Zonotope::origin(n);
        check(Z.dim() == n && Z.generators().shape() == std::vector<int64_t>({n, 0}),
              b + ": " + std::to_string(n) + " dimensions, no generators");
        check(close(Z.center(), std::vector<double>(n, 0.0), 0), b + ": the center is 0");
    }
    const Zonotope Z = Zonotope::origin(2);
    check(Z.contains(test::column({0.0, 0.0}), "exact", 0.0), b + ": it contains the origin");
    check(!Z.contains(test::column({0.0, 1e-3}), "exact", 0.0), b + ": and nothing else");
}

/// A dimension below 1 is refused; with the checks off the set is the same.
void checks_and_their_switch() {
    check(test::raises([] { Zonotope::origin(0); }, Error::wrong_value), "origin: dimension 0");
    check(test::raises([] { Zonotope::origin(-1); }, Error::wrong_value), "origin: dimension -1");
    ChecksScope off(false);
    check(close(Zonotope::origin(2).center(), std::vector<double>{0, 0}, 0),
          "origin: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(origin_is_the_point_zero);
    checks_and_their_switch();
    return test::finish("zonotope origin");
}
