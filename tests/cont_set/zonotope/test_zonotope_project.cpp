// test_zonotope_project - zonotope project: the selected dimensions, exactly
//
// CORA: contSet/@zonotope/project.m

#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

namespace {

/// The projection's support function in d is the original's in the direction d placed at the
/// kept dimensions, in the order they were asked for.
void project_selects_and_orders_dimensions(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope Z = Zonotope::generate_random(3, 4, rng);
    const Set P = Z.project({2, 0});
    check(P->dim() == 2, b + ": two dimensions remain");
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        check(close(test::support(*P, d), test::support(Z, {d[1], 0.0, d[0]}), 1e-10),
              b + ": support of the projection");
    }
}

void project_rejects_unknown_dimensions(const std::string &b) {
    cora::Rng rng(5);
    const Zonotope Z = Zonotope::generate_random(3, 2, rng);
    check(test::throws([&] { Z.project({0, 3}); }), b + ": dimension 3 does not exist");
    check(test::throws([&] { Z.project({-1, 0}); }), b + ": dimensions are not negative");
}

/// No dimension or one outside the set is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope S = Zonotope(Tensor({1.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(test::raises([&] { S.project({}); }, Error::wrong_value), "project: no dimension");
    check(test::raises([&] { S.project({2}); }, Error::wrong_value), "project: dimension 2 of 2");
    const Tensor on = S.project({1}).get().center();
    ChecksScope off(false);
    check(test::close(S.project({1}).get().center(), on, 0),
          "project: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        project_selects_and_orders_dimensions(b);
        project_rejects_unknown_dimensions(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope project");
}
