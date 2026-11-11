// test_zonotope_reduce_pca - Zonotope reduce_pca: the box of the generators in their principal axes
//
// More nonzero generators than dimensions become n generators of the box in the axes of G G';
// with at most n nonzero generators the zonotope stays. A hand case on the diagonal, the stated
// property (the result holds the zonotope, in n generators) on random zonotopes, and the batch.
//
// CORA: contSet/@zonotope/reduce.m
// CORA: contSet/@zonotope/private/priv_reducePCA.m
// PROPERTY: contSet/@zonotope/private/priv_reducePCA.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: n generators, the same center, and every support function at least that of Z.
void reduce_property(const std::string &b) {
    Rng rng(17);
    for (int k = 0; k < 10; ++k) {
        const Zonotope Z = Zonotope::generate_random(rng, 3, std::nullopt, 8);
        const Zonotope R = Z.reduce_pca();
        check(R.generators().shape() == std::vector<int64_t>({3, 3}), b + ": n generators");
        check(close(R.center(), Z.center()), b + ": the center stays");
        for (int d = 0; d < 10; ++d) {
            std::vector<double> v(3);
            rng.normal(v.data(), 3, 1.0);
            const Tensor dir = Tensor::from_data(v, {3, 1});
            check(R.support_function(dir).item() >= Z.support_function(dir).item() - 1e-9,
                  b + ": the result holds the zonotope");
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // two generators in the plane stay as they are
        const Zonotope two(Tensor({1.0, 2.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
        check(close(two.reduce_pca().generators(), two.generators()), b + ": n generators stay");
        // axis-aligned generators: the box, in the axes (the signs of the axes are free)
        const Zonotope diag(Tensor({0.0, 0.0}), Tensor({{3.0, 1.0, 0.0}, {0.0, 0.0, 2.0}}));
        const Zonotope R = diag.reduce_pca();
        check(close(R.generators().abs().sum(-1, true), Tensor({{4.0}, {2.0}}), 1e-12),
              b + ": the box of the axis-aligned generators");
        reduce_property(b);
    });
    ChecksScope off(false);
    check(Zonotope(Tensor({0.0}), Tensor({{1.0}})).reduce_pca().dim() == 1,
          "reduce_pca: valid with the checks off");
    return test::finish("zonotope reduce_pca");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
