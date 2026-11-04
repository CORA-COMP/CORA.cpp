// specification - the oracle's check of a specification on a box or a zonotope (CORALean)
//
// See also: lean/request.h

#include "cora/global/oracle/specification.h"

#include "cora/global/oracle/request.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The specification's set: the polytope A x <= b, in the dtype of A.
Wire aux_polytope(const Tensor &A, const Tensor &b) {
    Json set = Json::object();
    set.set("A", A.to_json()).set("b", b.to_json());
    return Wire(set, A.dtype());
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool check_specification(const std::string &kind, const Tensor &A, const Tensor &b,
                         const Interval &I) {
    return ask("specification.check", {{"set", aux_polytope(A, b)}, {"kind", kind}, {"I", I}})
        .get<bool>("value");
}

bool check_specification(const std::string &kind, const Tensor &A, const Tensor &b,
                         const Zonotope &Z) {
    return ask("specification.check", {{"set", aux_polytope(A, b)}, {"kind", kind}, {"Z", Z}})
        .get<bool>("value");
}

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
