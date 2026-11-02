// convert - what crosses between Python and C++: arrays, sets, option structs, errors
//
// Arrays: a torch tensor runs on the libtorch backend (when the package is built against torch,
// CORACPP_PYTHON_TORCH), a numpy array on Eigen; a vector is 1-D in Python, a column (..., n, 1)
// in C++. Sets: any cora set, a Python subclass of ContSet included, becomes a cora::Set and comes
// back as the object of its class. Option structs: a problem and an algorithm are filled from
// keyword arguments through their field lists; a keyword that fits none is refused with the
// closest name. Only conversion lives here, no computation.
//
// Syntax:   Tensor t = tensor_of(obj);   py::object o = vector(t);   Set S = set_of(obj);
//           ReachProblem p = fill<ReachProblem>(kwargs);   auto alg =
//           algorithm<alg::LinearReach>("standard", kwargs);
// See also: bindings.cpp, cora/global/options.h

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/eigen.h"
#include "cora/global/options.h"

#include <algorithm>
#include <optional>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <variant>
#ifdef CORACPP_PYTHON_TORCH
#include "cora/global/backend/torch.h"
#include <torch/extension.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::python {

namespace py = pybind11;

// Arrays ------------------------------------------------------------------------------------------

/// A Python vector (..., n) as a column (..., n, 1).
#ifdef CORACPP_PYTHON_TORCH
inline Tensor column(const torch::Tensor &v) { return from_torch(v.unsqueeze(-1)); }
#endif
inline Tensor column(const Eigen::VectorXd &v) { return from_eigen(v); }

/// A column (..., n, 1) as a Python vector (..., n), in the array type of its backend.
inline py::object vector(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t).squeeze(-1));
#endif
    if (is_eigen(t)) return py::cast(Eigen::VectorXd(to_eigen(t).col(0)));
    CORA_FAIL(Error::not_supported, "cora: a tensor of a backend Python cannot take (lean?); "
                                    "use the eigen or torch backend");
}

/// A matrix in the array type of its backend.
inline py::object matrix(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t));
#endif
    if (is_eigen(t)) return py::cast(to_eigen(t));
    CORA_FAIL(Error::not_supported, "cora: a tensor of a backend Python cannot take (lean?); "
                                    "use the eigen or torch backend");
}

/// The bounds of an interval: a column bound is a vector (..., n), a matrix bound stays.
inline py::object bounds(const Tensor &t) { return t.shape().back() == 1 ? vector(t) : matrix(t); }

/// A (..., 1, 1) result as a scalar per batch element: a torch tensor (...) or a float.
inline py::object scalar(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t).squeeze(-1).squeeze(-1));
#endif
    if (is_eigen(t)) return py::cast(to_eigen(t)(0, 0));
    CORA_FAIL(Error::not_supported, "cora: a tensor of a backend Python cannot take (lean?)");
}

/// A torch tensor or numpy array (or a list of numbers) as a Tensor on its backend, 1-D as a
/// column; nothing for anything else.
inline std::optional<Tensor> as_tensor(const py::handle &o) {
#ifdef CORACPP_PYTHON_TORCH
    py::detail::make_caster<torch::Tensor> torch_caster;
    if (torch_caster.load(o, /*convert=*/false)) {
        const torch::Tensor t = py::cast<torch::Tensor>(o);
        return t.dim() == 1 ? column(t) : from_torch(t);
    }
#endif
    if (py::isinstance<py::array>(o) || py::isinstance<py::list>(o) ||
        py::isinstance<py::tuple>(o)) {
        const py::array a = py::array_t<double, py::array::forcecast>::ensure(o);
        if (!a) return std::nullopt;
        if (a.ndim() == 1) return column(py::cast<Eigen::VectorXd>(a));
        if (a.ndim() == 2) return from_eigen(py::cast<Eigen::MatrixXd>(a));
    }
    return std::nullopt;
}

/// As as_tensor, an Error if it is not an array.
inline Tensor tensor_of(const py::handle &o, const char *what) {
    if (std::optional<Tensor> t = as_tensor(o)) return *t;
    CORA_FAIL(Error::wrong_value,
              std::string("cora: ") + what + " must be a numpy array or a torch tensor");
}

/// A vector (..., n) from Python as a column (..., n, 1): a torch tensor of any batch, or a 1-D
/// numpy array or list.
inline Tensor vector_of(const py::handle &o, const char *what) {
#ifdef CORACPP_PYTHON_TORCH
    py::detail::make_caster<torch::Tensor> torch_caster;
    if (torch_caster.load(o, /*convert=*/false)) return column(py::cast<torch::Tensor>(o));
#endif
    if (py::isinstance<py::array>(o) || py::isinstance<py::list>(o) || py::isinstance<py::tuple>(o)) {
        const py::array a = py::array_t<double, py::array::forcecast>::ensure(o);
        if (a && a.ndim() == 1) return column(py::cast<Eigen::VectorXd>(a));
        if (a && a.ndim() == 0) return column(Eigen::VectorXd::Constant(1, py::cast<double>(o)));
    }
    CORA_FAIL(Error::wrong_value, std::string("cora: ") + what +
                                      " must be a vector: a torch tensor (..., n) or a 1-D numpy array");
}

/// Whether the GIL can still be taken: not once the interpreter is finalized (at process exit,
/// when static objects holding Python objects are destroyed).
inline bool python_alive() { return Py_IsInitialized() != 0; }

// Sets --------------------------------------------------------------------------------------------

/// Any cora set as a Set: a copy of a library set, the object itself for a Python subclass.
inline Set set_of(const py::handle &o) {
    if (py::isinstance<Zonotope>(o)) return Set(py::cast<Zonotope>(o));
    if (py::isinstance<Interval>(o)) return Set(py::cast<Interval>(o));
    if (py::isinstance<ContSet>(o)) {
        // the Python object owns the C++ part; the Set keeps the object alive
        const ContSet *S = py::cast<const ContSet *>(o);
        py::object keep = py::reinterpret_borrow<py::object>(o);
        auto release = [keep](const ContSet *) mutable {
            if (!python_alive()) {
                keep.release();  // the interpreter is gone: leak, as Python does at exit
                return;
            }
            py::gil_scoped_acquire gil;
            keep = py::object();
        };
        return Set(std::shared_ptr<const ContSet>(S, release));
    }
    CORA_FAIL(Error::wrong_value, "cora: expected a set (Zonotope, Interval, a ContSet), got " +
                                      std::string(py::str(py::type::of(o).attr("__name__"))));
}

/// A Set as the Python object of its class.
inline py::object object_of(const Set &S) {
    switch (S.class_id()) {
    case ClassId::zonotope: return py::cast(S.as<Zonotope>());
    case ClassId::interval: return py::cast(S.as<Interval>());
    default: return py::cast(&S.get(), py::return_value_policy::reference);
    }
}

// Option structs ----------------------------------------------------------------------------------

/// A value of a field from Python.
template <class T>
T value_of(const py::handle &o, const char *name) {
    if constexpr (std::is_same_v<T, Tensor>) {
        return tensor_of(o, name);
    } else if constexpr (std::is_same_v<T, std::optional<Tensor>>) {
        if (o.is_none()) return std::nullopt;
        return tensor_of(o, name);
    } else if constexpr (std::is_same_v<T, std::optional<Set>>) {
        if (o.is_none()) return std::nullopt;
        return set_of(o);
    } else {
        return py::cast<T>(o);
    }
}

/// S with the fields given in kwargs; the keys used are removed from kwargs.
template <class S>
S fill(S s, py::dict &kwargs) {
    std::apply(
        [&](const auto &...f) {
            (([&] {
                 if (!kwargs.contains(f.name)) return;
                 using T = std::decay_t<decltype(s.*(f.member))>;
                 s.*(f.member) = value_of<T>(kwargs[f.name], f.name);
                 PyDict_DelItemString(kwargs.ptr(), f.name);
             }()),
             ...);
        },
        S::fields());
    return s;
}

/// The names of the alternatives of an algorithm variant.
template <class Variant, std::size_t... I>
std::vector<std::string> algorithm_names(std::index_sequence<I...>) {
    return {std::variant_alternative_t<I, Variant>::name...};
}
template <class Variant>
std::vector<std::string> algorithm_names() {
    return algorithm_names<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>());
}

/// The fields of every alternative of an algorithm variant.
template <class Variant, std::size_t... I>
std::vector<std::string> algorithm_fields(std::index_sequence<I...>) {
    std::vector<std::string> out;
    (([&] {
         for (const std::string &f : field_names<std::variant_alternative_t<I, Variant>>())
             out.push_back(f);
     }()),
     ...);
    return out;
}

/// The alternative of the variant named `name`, filled from kwargs; nothing if none has the name.
template <class Variant, std::size_t I = 0>
std::optional<Variant> algorithm(const std::string &name, py::dict &kwargs) {
    if constexpr (I == std::variant_size_v<Variant>) {
        return std::nullopt;
    } else {
        using A = std::variant_alternative_t<I, Variant>;
        if (name == A::name) return Variant(fill(A{}, kwargs));
        return algorithm<Variant, I + 1>(name, kwargs);
    }
}

/// An Error for keywords that no field took: an option of another algorithm is named so, any
/// other keyword gets the closest known name.
inline void refuse_leftovers(const py::dict &kwargs, const std::vector<std::string> &known,
                             const std::string &where, const std::vector<std::string> &others = {},
                             const std::string &alg = "") {
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        if (std::find(others.begin(), others.end(), key) != others.end())
            CORA_FAIL(Error::redundant_name_value_pair,
                      where + ": '" + key + "' is not an option of the algorithm '" + alg + "'");
        CORA_FAIL(Error::redundant_name_value_pair,
                  where + ": unknown keyword '" + key + "'" + did_you_mean(key, known));
    }
}

/// The message of an unknown algorithm name.
inline std::string unknown_algorithm(const std::string &name, const std::vector<std::string> &known,
                                     const std::string &where) {
    std::string list;
    for (const std::string &k : known) list += (list.empty() ? "'" : ", '") + k + "'";
    return where + ": unknown alg '" + name + "'" + did_you_mean(name, known) + "; use " + list;
}

} // namespace cora::python

// ---------------------------------------  END OF CODE  ---------------------------------------- //
