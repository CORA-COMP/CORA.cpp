// test_interval_center - interval center, rad and dim, for boxes and for interval matrices

#include "contSet/interval/interval.h"
#include "global/rng.h"
#include "testing.h"

using namespace cora::ct;
using test::check;
using test::close;
using test::column;

namespace {

void basics(const std::string &b) {
    const Interval I(column({-1.0, 0.0, 2.0}), column({3.0, 0.5, 2.0}));
    check(I.dim() == 3, b + ": dim");
    check(close(I.center(), std::vector<double>{1.0, 0.25, 2.0}), b + ": center");
    check(close(I.rad(), std::vector<double>{2.0, 0.25, 0.0}), b + ": radius");
    check(close(I.interval().inf, I.inf) && close(I.interval().sup, I.sup), b + ": its own hull");
}

/// As in CORA an interval can hold matrices; centre and radius are then elementwise.
void bounds_can_be_matrices(const std::string &b) {
    const Interval M(Tensor({{-1.0, 0.0}, {2.0, -3.0}}), Tensor({{1.0, 4.0}, {2.0, 0.0}}));
    check(close(M.center(), std::vector<double>{0.0, 2.0, 2.0, -1.5}), b + ": matrix centre");
    check(close(M.rad(), std::vector<double>{1.0, 2.0, 0.0, 1.5}), b + ": matrix radius");
    check(M.center().shape() == std::vector<int64_t>({2, 2}), b + ": matrix shape");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        basics(b);
        bounds_can_be_matrices(b);
    });
    return test::finish("interval center");
}
