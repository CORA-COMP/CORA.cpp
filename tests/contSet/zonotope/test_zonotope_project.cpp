// test_zonotope_project - zonotope project: the selected dimensions, exactly

#include "contSet/zonotope/zonotope.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;

namespace {

/// The projection's support function in d is the original's in the direction d placed at the
/// kept dimensions, in the order they were asked for.
void project_selects_and_orders_dimensions(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope Z = Zonotope::generateRandom(3, 4, rng);
    const std::unique_ptr<ContSet> P = Z.project({2, 0});
    check(P->dim() == 2, b + ": two dimensions remain");
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        check(close(test::support(*P, d), test::support(Z, {d[1], 0.0, d[0]}), 1e-10),
              b + ": support of the projection");
    }
}

void project_rejects_unknown_dimensions(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope Z = Zonotope::generateRandom(3, 2, rng);
    check(test::throws([&] { Z.project({0, 3}); }), b + ": dimension 3 does not exist");
    check(test::throws([&] { Z.project({-1, 0}); }), b + ": dimensions are not negative");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        project_selects_and_orders_dimensions(b);
        project_rejects_unknown_dimensions(b);
    });
    return test::finish("zonotope project");
}
