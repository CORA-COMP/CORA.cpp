// test_zonotope_split - zonotope split: pieces of a zonotope that together cover it
//
// CORA: contSet/@zonotope/split.m
// PROPERTY: contSet/@zonotope/split.m

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// CORA's unit test: the zonotope [-4 -3 -2 -1; 1 2 3 4] (center (-4, 1)), cut along the box
/// coordinates, a direction and a halfspace.
Zonotope example() {
    return {column({-4.0, 1.0}), Tensor({{-3.0, -2.0, -1.0}, {2.0, 3.0, 4.0}})};
}

void split_by_coordinates(const std::string &b) {
    const auto all = example().split();
    check(all.size() == 2, b + ": one pair per coordinate");
    check(close(all[0].first.center(), std::vector<double>{-7, 1}, 1e-14) &&
              close(all[0].second.center(), std::vector<double>{-1, 1}, 1e-14) &&
              close(all[0].first.generators(), std::vector<double>{3, 0, 0, 9}, 1e-14),
          b + ": coordinate 0 of the box (centers -4 -+ 3, generators 3 and 9)");
    const auto [lower, upper] = example().split(1);
    check(close(lower.center(), std::vector<double>{-4, -3.5}, 1e-14) &&
              close(upper.center(), std::vector<double>{-4, 5.5}, 1e-14) &&
              close(upper.generators(), std::vector<double>{6, 0, 0, 4.5}, 1e-14),
          b + ": coordinate 1");
    // a flat coordinate: CORA indexes the nonzero generators of the box and breaks; this indexes
    // coordinates, and both halves of a flat one are the box
    const Zonotope flat(column({0.0, 0.0}), Tensor({{1.0}, {0.0}}));
    const auto [a, c] = flat.split(1);
    check(close(a.center(), c.center(), 0) && close(a.generators(), std::vector<double>{1, 0, 0, 0}, 0),
          b + ": a flat coordinate is not split");
}

void split_along_a_generator(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{2.0, 0.0}, {0.0, 4.0}}));
    const auto [lower, upper] = Z.split(1, 2);
    check(close(lower.center(), std::vector<double>{0, -2}, 0) &&
              close(upper.center(), std::vector<double>{0, 2}, 0) &&
              close(upper.generators(), std::vector<double>{2, 0, 0, 2}, 0),
          b + ": generator 1, no reduction needed");
    // a reduction to order 1 leaves two generators: the kept longest one and a box generator
    const Zonotope W(column({0.0, 0.0}), Tensor({{4.0, 0.0, 1.0}, {0.0, 3.0, 1.0}}));
    const auto [l2, u2] = W.split(0, 1);
    check(l2.generators().shape() == std::vector<int64_t>({2, 2}), b + ": reduced to order 1");
    check(close(u2.center(), std::vector<double>{2.5, 0}, 0),
          b + ": the box generator (5, 0) of the reduction is halved");
}

void split_by_a_direction(const std::string &b) {
    const auto [one, two] = example().split(column({1.0, 1.0}));
    check(close(one.center(), std::vector<double>{-5.25, -0.25}, 1e-14) &&
              close(two.center(), std::vector<double>{-2.75, 2.25}, 1e-14),
          b + ": the centers of CORA's example");
    check(close(one.generators(), std::vector<double>{1.25, -2.5, -2.5, -2.5, 1.25, 2.5, 2.5, 2.5}, 1e-14) &&
              close(two.generators(),
                    std::vector<double>{-1.25, -2.5, -2.5, -2.5, -1.25, 2.5, 2.5, 2.5}, 1e-14),
          b + ": the generators of CORA's example");
    // the plane normal to aux_dir minus its part along orig_dir: (1, 0) and (1, 1) give (0, 1)
    const auto [p, q] = example().split(column({1.0, 0.0}), column({1.0, 1.0}));
    const auto [r, s] = example().split(column({0.0, 1.0}));
    check(close(p.center(), r.center(), 1e-14) && close(q.generators(), s.generators(), 1e-14),
          b + ": the perpendicular direction");
}

void split_by_a_halfspace(const std::string &b) {
    const auto pieces = example().split(column({1.0, -1.0}), -2.0);
    check(pieces.size() == 2, b + ": the plane meets the set");
    check(close(pieces[0].center(), std::vector<double>{-7, 4}, 1e-14) &&
              close(pieces[1].center(), std::vector<double>{0.5, -3.5}, 1e-14) &&
              close(pieces[0].generators(),
                    std::vector<double>{4.5, -0.5, 0.5, 1.5, -4.5, -0.5, 0.5, 1.5}, 1e-14),
          b + ": CORA's example");
    check(example().split(column({1.0, 0.0}), 100.0).size() == 1, b + ": a plane that misses: the set");
    check(example().split(column({1.0, 0.0}), -100.0).size() == 1, b + ": a plane beyond the other side");
}

/// Stated property: every point of Z is in one of the pieces (all kinds of cut), and the pieces of
/// a coordinate cut cover the zonotope's box part along it.
void pieces_cover_the_set(const std::string &b) {
    cora::Rng rng(91);
    const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const Tensor points = Z.rand_point(40, rng);
    const Tensor normal = column({1.0, -2.0, 0.5}), orig = column({0.0, 1.0, 0.0});
    const std::vector<std::pair<Zonotope, Zonotope>> cuts{
        Z.split(1, 4), Z.split(normal), Z.split(orig, column({1.0, 1.0, 0.0}))};
    for (const auto &[first, second] : cuts)
        for (int k = 0; k < 40; ++k) {
            const Tensor p = points.slice(-1, k, k + 1);
            check(first.contains(p, "exact", 1e-9) || second.contains(p, "exact", 1e-9),
                  b + ": a point of Z is in neither piece");
        }
    const Tensor lift = column({1.0, 1.0, 1.0});
    const auto planes = Z.split(lift, Z.center().matrix_transpose().matmul(lift).item() + 0.1);
    check(planes.size() == 2, b + ": a plane near the center cuts the set");
    for (int k = 0; k < 40; ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        check(planes[0].contains(p, "exact", 1e-9) || planes[1].contains(p, "exact", 1e-9),
              b + ": a point of Z is in neither piece of the halfspace cut");
    }
    // the box pieces cover the box: the hull of the halves is the box
    const Interval box = Z.interval();
    for (int64_t i = 0; i < 3; ++i) {
        const auto [lo, hi] = Z.split(i);
        const Interval a = lo.interval(), c = hi.interval();
        check(close(a.inf().minimum(c.inf()), box.inf(), 1e-12) &&
                  close(a.sup().maximum(c.sup()), box.sup(), 1e-12),
              b + ": and cover it");
    }
}

/// Wrong coordinates, directions, orders and batches are refused; with the checks off the pieces
/// are the same.
void checks_and_their_switch() {
    const Zonotope Z = example();
    check(test::raises([&] { Z.split(2); }, Error::wrong_value), "split: a coordinate beyond");
    check(test::raises([&] { Z.split(-1); }, Error::wrong_value), "split: a negative coordinate");
    check(test::raises([&] { Z.split(0, 0); }, Error::wrong_value), "split: an order below 1");
    check(test::raises([&] { Z.split(5, 1); }, Error::wrong_value), "split: a generator beyond");
    check(test::raises([&] { Z.split(column({0.0, 0.0})); }, Error::wrong_value), "split: dir zero");
    check(test::raises([&] { Z.split(column({1.0, 0.0, 0.0})); }, Error::dimension_mismatch),
          "split: dir of the wrong size");
    check(test::raises([&] { Z.split(column({1.0, 2.0}), column({2.0, 4.0})); }, Error::wrong_value),
          "split: parallel directions");
    check(test::raises([&] { Z.split(column({1.0, 0.0, 0.0}), 1.0); }, Error::dimension_mismatch),
          "split: a of the wrong size");
    check(test::raises([&] { Z.split(column({0.0, 0.0}), 1.0); }, Error::wrong_value), "split: a zero");
    const Zonotope batch = Zonotope::stack({Z, Z});
    check(test::raises([&] { batch.split(column({1.0, 1.0})); }, Error::not_supported), "split: batch dir");
    check(test::raises([&] { batch.split(column({1.0, 1.0}), 0.0); }, Error::not_supported),
          "split: batch halfspace");
    check(batch.split(0).first.center().shape() == std::vector<int64_t>({2, 2, 1}),
          "split: a batch splits by coordinate");
    ChecksScope off(false);
    check(close(Z.split(0).first.center(), std::vector<double>{-7, 1}, 1e-14), "split: checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        split_by_coordinates(b);
        split_along_a_generator(b);
        split_by_a_direction(b);
        split_by_a_halfspace(b);
        pieces_cover_the_set(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope split");
}
