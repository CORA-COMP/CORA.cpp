// test_interval_kron - interval kron: the Kronecker product of interval matrices
//
// CORA: contSet/@interval/kron.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void kron_is_the_hull_of_the_corner_products(const std::string &b) {
    // I = [[1, 2], [-1, 1]] (a row), J = ([0, 1], [2, 3]) (a column)
    const Interval I(Tensor::from_data({1, -1}, {1, 2}), Tensor::from_data({2, 1}, {1, 2}));
    const Interval J(Tensor::from_data({0, 2}, {2, 1}), Tensor::from_data({1, 3}, {2, 1}));
    const Interval P = I.kron(J);
    check(P.inf().shape() == std::vector<int64_t>({2, 2}), b + ": the shape (m p, n q)");
    check(close(P.inf(), std::vector<double>{0, -1, 2, -3}) &&
              close(P.sup(), std::vector<double>{2, 1, 6, 3}),
          b + ": block (i, j) is the product of the entries");
    // a number first: [[1, -1]] kron J, blocks 1 * J and -1 * J
    const Interval Q = Interval::kron(Tensor::from_data({1, -1}, {1, 2}), J);
    check(close(Q.inf(), std::vector<double>{0, -1, 2, -3}) &&
              close(Q.sup(), std::vector<double>{1, 0, 3, -2}),
          b + ": a numeric left factor");
    const Interval R = I.kron(Tensor::from_data({1, -1}, {2, 1}));
    check(close(R.inf(), std::vector<double>{1, -1, -2, -1}) &&
              close(R.sup(), std::vector<double>{2, 1, -1, 1}),
          b + ": a numeric right factor");
    const Interval S = Interval::stack({I, I}).kron(J);
    check(S.inf().shape() == std::vector<int64_t>({2, 2, 2}) &&
              close(S.sup().slice(0, 1, 2).reshape({2, 2}), P.sup()),
          b + ": a batch");
}

/// A numeric factor that is not a matrix is refused; the checks off give the same.
void checks_and_their_switch() {
    const Interval I(Tensor::from_data({1, -1}, {1, 2}), Tensor::from_data({2, 1}, {1, 2}));
    check(test::raises([&] { I.kron(Tensor::from_data({1.0, 2.0}, {2})); },
                       Error::dimension_mismatch),
          "kron: a flat right factor");
    check(test::raises([&] { Interval::kron(Tensor::from_data({1.0, 2.0}, {2}), I); },
                       Error::dimension_mismatch),
          "kron: a flat left factor");
    const Tensor on = I.kron(I).sup();
    ChecksScope off(false);
    check(close(I.kron(I).sup(), on, 0), "kron: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(kron_is_the_hull_of_the_corner_products);
    checks_and_their_switch();
    return test::finish("interval kron");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
