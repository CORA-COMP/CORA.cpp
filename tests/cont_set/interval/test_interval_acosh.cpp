// test_interval_acosh - interval acosh: the inverse hyperbolic cosine of a box
//
// CORA: contSet/@interval/acosh.m
// PROPERTY: contSet/@interval/acosh.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <algorithm>
#include <limits>

using namespace cora;
using test::check;
using test::close;
using test::vec;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A box of n dimensions with bounds drawn from [low, high].
Interval aux_sample(cora::Rng &rng, int n, double low, double high) {
    std::vector<double> a(n), b(n);
    rng.uniform(a.data(), n, low, high);
    rng.uniform(b.data(), n, low, high);
    for (int i = 0; i < n; ++i)
        if (a[i] > b[i]) std::swap(a[i], b[i]);
    return {vec(a), vec(b)};
}

/// The bounds map as the function does; a batch is mapped member by member.
void acosh_maps_the_bounds(const std::string &b) {
    const Interval I(vec({1.0, 2.0}), vec({2.0, 10.0}));
    const Interval R = I.acosh();
    check(close(R.inf(), std::vector<double>{0.0, 1.3169578969248166}) &&
              close(R.sup(), std::vector<double>{1.3169578969248166, 2.993222846126381}),
          b + ": acosh of the bounds");
    const Interval S = Interval::stack({I, I}).acosh();
    check(S.inf().shape() == std::vector<int64_t>({2, 2}), b + ": a batch has a batch shape");
    check(close(S.sup().slice(0, 1, 2).reshape({2}), R.sup()), b + ": a batch member");
}

/// The image of every sampled point of the box lies in the result.
void acosh_encloses_the_image(const std::string &b) {
    cora::Rng rng(34);
    for (int trial = 0; trial < 20; ++trial) {
        const Interval I = aux_sample(rng, 3, 1, 9);
        const Interval R = I.acosh();
        const Tensor x = I.rand_point(20, rng).matrix_transpose();
        bool inside = true;
        for (int k = 0; k < 20; ++k)
            inside = inside && R.contains(x.slice(-1, k, k + 1).squeeze(-1).acosh());
        check(inside, b + ": acosh of a sample is in the result");
    }
}

/// A bound outside the domain is refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    check(test::raises([] { Interval(vec({0.5, 2.0}), vec({2.0, 3.0})).acosh(); },
                       Error::out_of_domain),
          "acosh: a bound outside the domain");
    const Interval I(vec({1.0, 2.0}), vec({2.0, 10.0}));
    const Tensor on = I.acosh().sup();
    ChecksScope off(false);
    check(close(I.acosh().sup(), on, 0), "acosh: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        acosh_maps_the_bounds(b);
        acosh_encloses_the_image(b);
    });
    checks_and_their_switch();
    return test::finish("interval acosh");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
