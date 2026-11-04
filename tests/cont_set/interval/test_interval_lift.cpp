// test_interval_lift - interval lift: a box placed into a higher-dimensional space
//
// CORA: contSet/@interval/lift_.m

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

#include <limits>

using namespace cora;
using test::check;
using test::column;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

void lift_places_the_coordinates(const std::string &b) {
    const Interval I(column({1.0, 3.0}), column({2.0, 4.0}));
    const Interval R = I.lift(4, {2, 0});
    check(R.inf().data() == std::vector<double>({3.0, -kInf, 1.0, -kInf}) &&
              R.sup().data() == std::vector<double>({4.0, kInf, 2.0, kInf}),
          b + ": coordinate dims[j] is coordinate j of the box, the rest unbounded");
    check(I.lift(2, {0, 1}).inf().data() == I.inf().data(), b + ": N = dim and the identity");
    check(I.lift(3, {1, 1}).inf().data() == std::vector<double>({-kInf, 3.0, -kInf}),
          b + ": a repeated coordinate keeps the last, as CORA");
    const Interval S = Interval::stack({I, I}).lift(3, {1, 2});
    check(S.inf().shape() == std::vector<int64_t>({2, 3, 1}) &&
              S.sup().data() == std::vector<double>({kInf, 2.0, 4.0, kInf, 2.0, 4.0}),
          b + ": a batch");
}

/// A space that is too small, a wrong number or a range of coordinates, and an empty interval are
/// refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(column({1.0, 3.0}), column({2.0, 4.0}));
    check(test::raises([&] { I.lift(1, {0, 0}); }, Error::wrong_value),
          "lift: N below the dimension");
    check(test::raises([&] { I.lift(3, {0}); }, Error::wrong_value),
          "lift: one coordinate for two");
    check(test::raises([&] { I.lift(3, {0, 3}); }, Error::wrong_value), "lift: coordinate 3 of 3");
    check(test::raises([&] { I.lift(3, {-1, 0}); }, Error::wrong_value),
          "lift: a negative coordinate");
    check(test::raises([&] { Interval::empty(2).lift(3, {0, 1}); }, Error::not_supported),
          "lift: an empty interval");
    const std::vector<double> on = I.lift(3, {2, 0}).inf().data();
    ChecksScope off(false);
    check(I.lift(3, {2, 0}).inf().data() == on, "lift: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(lift_places_the_coordinates);
    checks_and_their_switch();
    return test::finish("interval lift");
}
