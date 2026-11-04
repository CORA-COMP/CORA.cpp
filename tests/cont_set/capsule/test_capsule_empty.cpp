// test_capsule_empty - Capsule::empty: the capsule without points
//
// CORA: contSet/@capsule/empty.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// The empty capsule has the radius -1, no point, and a center without columns.
void empty_known(const std::string &b) {
    const Capsule C = Capsule::empty(3);
    check(C.dim() == 3 && C.is_empty(), b + ": dimension and emptiness");
    check(close(C.r(), {-1}, 0) && close(C.c(), {0, 0, 0}, 0), b + ": the radius -1 at the origin");
    check(C.center().shape() == Shape{3, 0}, b + ": the center has no column");
    check(Capsule::empty().dim() == 0, b + ": the dimension defaults to 0");
    check(!C.contains(column({0.0, 0.0, 0.0})), b + ": it contains no point");
    check(!Capsule(column({0.0, 0.0, 0.0})).is_empty(), b + ": a point is not empty");
    check(C.interval().is_empty(), b + ": its box is empty");
}

/// A batch is empty if every member is; a mixed batch is not.
void empty_batch(const std::string &b) {
    const Tensor c = Tensor::zeros({2, 2, 1});
    // the checked constructor refuses the radius -1, so these are built with the checks off
    ChecksScope off(false);
    const Capsule all(c, c, Tensor::from_data({-1.0, -1.0}, {2, 1, 1}));
    const Capsule mixed(c, c, Tensor::from_data({-1.0, 1.0}, {2, 1, 1}));
    check(!mixed.is_empty(), b + ": a mixed batch is not empty");
    check(all.is_empty(), b + ": every member empty");
}

/// A negative dimension is refused; the same call with the checks off still builds an empty set.
void checks_and_their_switch() {
    check(test::raises([] { Capsule::empty(-1); }, Error::wrong_value),
          "empty: the dimension is nonnegative");
    ChecksScope off(false);
    check(Capsule::empty(2).is_empty(), "empty: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        empty_known(b);
        empty_batch(b);
    });
    checks_and_their_switch();
    return test::finish("capsule empty");
}
