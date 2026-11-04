// test_zonotope_project_high_dim - zonotope project_high_dim: a zonotope in a larger space
//
// CORA: contSet/@zonotope/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// State i goes to position dims[i], the others are 0; the identity keeps the set; a repeated
/// position keeps the last state; a batch embeds member by member.
void embedding_known(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.0}, {0.0, 3.0}}));
    const Zonotope R = Z.project_high_dim(4, {3, 1});
    check(close(R.center(), std::vector<double>{0, 2, 0, 1}, 0), b + ": the center");
    check(close(R.generators(), std::vector<double>{0, 0, 0, 3, 0, 0, 1, 0}, 0),
          b + ": the generators");
    check(close(Z.project_high_dim(2, {0, 1}).generators(), Z.generators(), 0),
          b + ": the identity");
    check(close(Z.project_high_dim(2, {1, 1}).center(), std::vector<double>{0, 2}, 0),
          b + ": a repeated position keeps the last state");
    const Zonotope batch = Zonotope::stack({Z, Z + column({1.0, 1.0})}).project_high_dim(3, {2, 0});
    check(batch.center().shape() == std::vector<int64_t>({2, 3, 1}) &&
              close(batch.center().slice(0, 1, 2).squeeze(0), std::vector<double>{3, 0, 2}, 0),
          b + ": a batch");
    const Zonotope P(column({1.0}), Tensor::zeros({1, 0}));
    check(P.project_high_dim(2, {1}).generators().shape() == std::vector<int64_t>({2, 0}),
          b + ": no generators");
}

/// Stated property: the support function along d is that of Z along the entries of d at dims.
void embedding_keeps_the_support(const std::string &b) {
    cora::Rng rng(71);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const std::vector<int64_t> dims{4, 0, 2};
    const Zonotope R = Z.project_high_dim(6, dims);
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 6);
        check(close(test::support(R, d), test::support(Z, {d[4], d[0], d[2]}), 1e-12),
              b + ": the support function");
    }
}

/// Too small a space, a wrong number of positions and a position outside are refused.
void checks_and_their_switch() {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0}, {0.0}}));
    check(test::raises([&] { Z.project_high_dim(1, {0, 0}); }, Error::wrong_value), "N below dim");
    check(test::raises([&] { Z.project_high_dim(3, {0}); }, Error::wrong_value),
          "one position short");
    check(test::raises([&] { Z.project_high_dim(3, {0, 3}); }, Error::wrong_value), "beyond N");
    check(test::raises([&] { Z.project_high_dim(3, {-1, 0}); }, Error::wrong_value), "negative");
    ChecksScope off(false);
    check(close(Z.project_high_dim(3, {2, 0}).center(), std::vector<double>{2, 0, 1}, 0),
          "project_high_dim: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        embedding_known(b);
        embedding_keeps_the_support(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope project_high_dim");
}
