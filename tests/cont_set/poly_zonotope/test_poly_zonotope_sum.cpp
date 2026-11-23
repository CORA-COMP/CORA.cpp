// test_poly_zonotope_sum - PolyZonotope sum: the sum over shared factors
//
// The points of the sum at the shared factors are the sum of the points of the summands, with
// the center of every summand added (CORA adds the first one's again, docs/cora-differences.md).
//
// Oracle debt: the request is in docs/oracle-requests.md (CORALean has the float definition).
//
// CORA: contSet/@polyZonotope/sum.m
// PROPERTY: contSet/@polyZonotope/sum.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "poly_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(21);
        const PolyZonotope P = test::random_poly(rng, 2), Q = test::random_poly(rng, 2),
                           R(vec({5.0, -1.0}), Tensor::from_data({1, 2}, {1, 2}),
                             Tensor::from_data({1, 1}, {1, 2}), {{1}}, {3});
        const PolyZonotope S = P.sum({Q, R});
        check(S.ids() == std::vector<int64_t>({1, 2, 3}), b + ": the union of the identifiers");
        // the points of the sum at shared factor values
        const std::map<int64_t, double> alpha = {{1, 0.3}, {2, -0.6}, {3, 0.9}};
        const std::vector<double> sp = test::point(S, alpha, {0.1, -0.2, 0.3, 0.4, 0.5}),
                                  pp = test::point(P, alpha, {0.1, -0.2}),
                                  qp = test::point(Q, alpha, {0.3, 0.4}),
                                  rp = test::point(R, alpha, {0.5});
        // the sum of the points of the summands
        std::vector<double> sum(2);
        for (int i = 0; i < 2; ++i)
            sum[std::size_t(i)] = pp[std::size_t(i)] + qp[std::size_t(i)] + rp[std::size_t(i)];
        check(close(Tensor::from_data(sp, {2}), Tensor::from_data(sum, {2}), 1e-12),
              b + ": the points add");
        check(close(S.center(), (P.center() + Q.center() + R.center()), 1e-12),
              b + ": every center is added once");
        check(P.sum({}).isequal(P), b + ": an empty list");
    });
    const PolyZonotope P = test::sample_poly();
    check(test::raises([&] { P.sum({PolyZonotope::origin(3)}); }, Error::dimension_mismatch),
          "sum: dimension");
    ChecksScope off(false);
    check(P.sum({P}).dim() == 2, "sum: valid with the checks off");
    return test::finish("poly_zonotope sum");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
