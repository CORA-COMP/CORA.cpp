// test_interval_partition - interval partition: a box cut into the boxes of an equal grid
//
// CORA: contSet/@interval/partition.m
// PROPERTY: contSet/@interval/partition.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// 2 x 3 pieces of [0, 4] x [0, 6] in the order of nested loops, the first coordinate slowest.
void partition_cuts_the_grid(const std::string &b) {
    const Interval I(column({0.0, 0.0}), column({4.0, 6.0}));
    const std::vector<Interval> P = I.partition(2);
    check(P.size() == 4, b + ": splits^n pieces");
    const std::vector<std::vector<double>> inf{{0, 0}, {0, 3}, {2, 0}, {2, 3}};
    const std::vector<std::vector<double>> sup{{2, 3}, {2, 6}, {4, 3}, {4, 6}};
    bool same = P.size() == 4;
    for (std::size_t k = 0; same && k < 4; ++k)
        same = close(P[k].inf(), inf[k]) && close(P[k].sup(), sup[k]);
    check(same, b + ": the pieces, the last coordinate fastest");
    check(I.partition(1).size() == 1 && close(I.partition(1)[0].sup(), I.sup()),
          b + ": one split is the box");
    const std::vector<Interval> B = Interval::stack({I, I}).partition(3);
    check(B.size() == 9 && B[4].inf().shape() == std::vector<int64_t>({2, 2, 1}),
          b + ": a batch has a batch shape");
}

/// The pieces cover the box: every sampled point is in a piece, and the pieces lie in the box.
void partition_covers_the_box(const std::string &b) {
    cora::Rng rng(71);
    for (int trial = 0; trial < 10; ++trial) {
        const Interval I = Interval::generate_random(rng, 3);
        const std::vector<Interval> P = I.partition(3 + trial % 3);
        const Tensor x = I.rand_point(40, rng);
        bool covered = true, inside = true;
        for (int k = 0; k < 40; ++k) {
            bool hit = false;
            for (const Interval &piece : P) hit = hit || piece.contains(x.slice(-1, k, k + 1));
            covered = covered && hit;
        }
        for (const Interval &piece : P)
            inside = inside && I.contains(piece.inf()) && I.contains(piece.sup());
        check(covered, b + ": the pieces cover the sample");
        check(inside, b + ": the pieces lie in the box");
    }
}

/// Neighbouring pieces touch or overlap, in whatever rounding the cuts have.
void partition_has_no_gaps(const std::string &b) {
    const Interval I(column({0.1}), column({0.7}));
    const std::vector<Interval> P = I.partition(7);
    bool joined = true;
    for (std::size_t k = 0; k + 1 < P.size(); ++k)
        joined = joined && P[k].sup().data()[0] >= P[k + 1].inf().data()[0];
    check(joined && close(P.front().inf(), I.inf(), 0) && close(P.back().sup(), I.sup(), 1e-15),
          b + ": no gap between neighbours, the ends are the bounds");
}

/// Splits below 1, a matrix and an empty box are refused, and so are too many pieces; with the
/// checks off the result is the same.
void checks_and_their_switch() {
    const Interval I(column({0.0, 1.0}), column({1.0, 2.0}));
    check(test::raises([&] { I.partition(0); }, Error::wrong_value), "partition: 0 splits");
    check(test::raises([&] { Interval::empty(2).partition(2); }, Error::not_supported),
          "partition: an empty box");
    check(test::raises([&] {
              Interval(Tensor::from_data({0, 0, 0, 0}, {2, 2}), Tensor::from_data({1, 1, 1, 1}, {2, 2}))
                  .partition(2);
          }, Error::not_supported),
          "partition: an interval matrix");
    check(test::raises([&] { I.partition(100000); }, Error::out_of_memory), "partition: 10^10 pieces");
    const std::vector<double> on = I.partition(2)[3].sup().data();
    ChecksScope off(false);
    check(I.partition(2)[3].sup().data() == on, "partition: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        partition_cuts_the_grid(b);
        partition_covers_the_box(b);
        partition_has_no_gaps(b);
    });
    checks_and_their_switch();
    return test::finish("interval partition");
}
