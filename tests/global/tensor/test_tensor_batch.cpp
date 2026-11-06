// test_tensor_batch - leading axes are batches on every backend, with the answers of one matrix
//
// A batched product, solve, exponential, factorization or reduction equals the same operation on
// each matrix of the batch, on eigen, lean and torch alike; batch axes broadcast as in numpy.
// Products are bit-identical on eigen and lean; libtorch picks its BLAS kernel for a batched and
// a single product per CPU, so on torch they agree to rounding only.
//
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/tensor/tensor.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Matrix b of a batch (B, n, m).
Tensor aux_member(const Tensor &x, int64_t b) { return x.slice(0, b, b + 1).squeeze(0); }

void batches_are_their_members(const std::string &name) {
    Rng rng(4);
    std::vector<double> va(3 * 4 * 4), vb(3 * 4 * 2);
    rng.normal(va.data(), va.size(), 1.0);
    rng.normal(vb.data(), vb.size(), 1.0);
    const Tensor A = Tensor::from_data(va, {3, 4, 4}) + Tensor::eye(4) * 4.0;
    const Tensor B = Tensor::from_data(vb, {3, 4, 2});
    const Tensor single = aux_member(B, 0);
    const double product_tol = name == "torch" ? 1e-12 : 0;
    for (int64_t b = 0; b < 3; ++b) {
        const Tensor Ab = aux_member(A, b), Bb = aux_member(B, b);
        const std::string at = name + ", member " + std::to_string(b);
        check(close(aux_member(A.matmul(B), b), Ab.matmul(Bb), product_tol), at + ": matmul");
        check(close(aux_member(A.matmul(single), b), Ab.matmul(single), product_tol),
              at + ": broadcast");
        check(close(aux_member(A.solve(B), b), Ab.solve(Bb), 1e-12), at + ": solve");
        check(close(aux_member(A.sum(-1, true), b), Ab.sum(-1, true), 0), at + ": sum");
        check(close(aux_member(A.diagonal(), b), Ab.diagonal(), 0), at + ": diagonal");
        // per-member decompositions
        if (name != "lean")
            check(close(aux_member((A * 0.1).expm(), b), (Ab * 0.1).expm(), 1e-12), at + ": expm");
        const auto [Q, R] = A.qr();
        check(close(aux_member(Q, b).matmul(aux_member(R, b)), Ab, 1e-12), at + ": qr");
    }
    check(Tensor::stack({single, single, single}).shape() == Shape({3, 4, 2}), name + ": stack");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    std::vector<std::string> all = test::backends();
    all.emplace_back("lean");
    for (const std::string &b : all) {
        const BackendScope scope(b);
        batches_are_their_members(b);
    }
    return test::finish("tensor batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
