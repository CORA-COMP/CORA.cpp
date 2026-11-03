// convert - what crosses between Python and C++: arrays, sets, option structs, errors
//
// Arrays: a torch tensor runs on the libtorch backend (when the package is built against torch,
// CORACPP_PYTHON_TORCH), a numpy array on Eigen over the array's own memory, and results come
// back as numpy arrays over the tensor's memory: no copy either way. A vector is 1-D in Python, a
// column (..., n, 1) in C++. Sets: any cora set, a Python subclass of ContSet included, becomes a
// cora::Set and comes back as the object of its class. Option structs: a problem and an algorithm
// are filled from keyword arguments through their field lists; a keyword that fits none is refused
// with the closest name. Only conversion lives here, no computation.
//
// Syntax:   Tensor t = tensor_of(obj);   py::object o = vector(t);   Set S = set_of(obj);
//           ReachProblem p = fill<ReachProblem>(kwargs);   auto alg =
//           algorithm<alg::LinearReach>("standard", kwargs);
// See also: bindings.cpp, cora/global/options.h

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/host.h"
#include "cora/global/backend/lean.h"
#include "cora/global/options.h"

#include <algorithm>
#include <optional>
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

/// A numpy array or torch tensor in a signature: the stubs (scripts/stubs.py) show it as `Array`.
class Array : public py::object {
public:
    PYBIND11_OBJECT_CVT(Array, py::object, any_object, share_object)
    Array() = default;
    static bool any_object(PyObject *) { return true; }
    static PyObject *share_object(PyObject *o) { return py::handle(o).inc_ref().ptr(); }
};

} // namespace cora::python

/// The name pybind11 prints for an Array in a signature.
namespace pybind11::detail {
template <>
struct handle_type_name<cora::python::Array> {
    static constexpr auto name = const_name("Array");
};
} // namespace pybind11::detail

namespace cora::python {

/// Whether the GIL can still be taken: not once the interpreter is finalized (at process exit,
/// when static objects holding Python objects are destroyed).
inline bool python_alive() { return Py_IsInitialized() != 0; }

/// What keeps a Python object alive while a tensor borrows its memory; released under the GIL.
inline std::shared_ptr<const void> keep_alive(const py::handle &o) {
    PyObject *p = o.ptr();
    Py_INCREF(p);
    return std::shared_ptr<const void>(p, [](const void *q) {
        if (!python_alive()) return; // the interpreter is gone: leak, as Python does at exit
        py::gil_scoped_acquire gil;
        Py_DECREF(static_cast<PyObject *>(const_cast<void *>(q)));
    });
}

/// A numpy array (or a list) as a host tensor over the array's memory, without a copy: on eigen,
/// or on lean under `with cora.backend("lean")` (so that CORA.cpp rounds as the CORALean oracle).
/// A copy is made only for an array that is not C-contiguous or not float64/float32 (lean:
/// float64). `column` makes a 1-D array a column (n, 1).
inline Tensor host_tensor(const py::handle &o, bool column) {
    const uint8_t id = backend().name() == "lean" ? backend_id::lean : backend_id::eigen;
    py::array a = py::array::ensure(o);
    if (!a) CORA_FAIL(Error::wrong_value, "cora: expected a numpy array");
    const bool f32 = a.dtype().is(py::dtype::of<float>()) && id == backend_id::eigen;
    const bool usable = f32 || a.dtype().is(py::dtype::of<double>());
    if (!usable || !(a.flags() & py::array::c_style))
        a = py::array_t < double, py::array::c_style | py::array::forcecast > ::ensure(o);
    Shape shape(std::vector<int64_t>(a.shape(), a.shape() + a.ndim()));
    if (column && a.ndim() == 1) shape.push_back(1);
    return host_borrow(a.data(), f32 ? DType::float32 : DType::float64, shape, keep_alive(a), id);
}

/// A host tensor as a numpy array of `shape` over the tensor's memory, without a copy: a capsule
/// keeps the tensor alive. Read-only unless the array is the memory's only holder.
inline py::array host_array(const Tensor &t, const Shape &shape) {
    const HostView v = host_view(t);
    const bool f32 = v.dtype == DType::float32;
    const py::ssize_t item = f32 ? sizeof(float) : sizeof(double);
    std::vector<py::ssize_t> dims(shape.begin(), shape.end()), strides(dims.size());
    py::ssize_t step = item;
    for (std::size_t d = dims.size(); d-- > 0;) {
        strides[d] = step;
        step *= dims[d];
    }
    py::capsule base(new Tensor::Ptr(t.ptr()),
                     [](void *p) { delete static_cast<Tensor::Ptr *>(p); });
    py::array out(f32 ? py::dtype::of<float>() : py::dtype::of<double>(), dims, strides, v.data,
                  base);
    if (!v.unique) out.attr("setflags")(py::arg("write") = false);
    return out;
}

/// The shape of a column (..., n, 1) as a vector (..., n).
inline Shape vector_shape(const Tensor &t) {
    Shape s;
    for (int64_t d = 0; d + 1 < t.ndim(); ++d) s.push_back(t.shape()[d]);
    return s;
}

/// A Python vector (..., n) as a column (..., n, 1).
#ifdef CORACPP_PYTHON_TORCH
inline Tensor column(const torch::Tensor &v) { return from_torch(v.unsqueeze(-1)); }
#endif

/// A column (..., n, 1) as a Python vector (..., n), in the array type of its backend.
inline Array vector(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t).squeeze(-1));
#endif
    if (is_host(t)) return host_array(t, vector_shape(t));
    CORA_FAIL(Error::not_supported, "cora: a tensor of a backend Python cannot take; "
                                    "use the eigen, torch or lean backend");
}

/// A matrix (or a batch) in the array type of its backend.
inline Array matrix(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t));
#endif
    if (is_host(t)) return host_array(t, t.shape());
    CORA_FAIL(Error::not_supported, "cora: a tensor of a backend Python cannot take; "
                                    "use the eigen, torch or lean backend");
}

/// The bounds of an interval: a column bound is a vector (..., n), a matrix bound stays.
inline Array bounds(const Tensor &t) { return t.shape().back() == 1 ? vector(t) : matrix(t); }

/// A (..., 1, 1) result as a scalar per batch element: a torch tensor (...), a float for one
/// matrix, else an array (...).
inline py::object scalar(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(t)) return py::cast(to_torch(t).squeeze(-1).squeeze(-1));
#endif
    if (t.numel() == 1) return py::cast(t.item());
    return host_array(t, vector_shape(t.squeeze(-1)));
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
        if (!a || a.ndim() == 0) return std::nullopt;
        return host_tensor(o, true);
    }
    return std::nullopt;
}

/// As as_tensor, an Error if it is not an array.
inline Tensor tensor_of(const py::handle &o, const char *what) {
    if (std::optional<Tensor> t = as_tensor(o)) return *t;
    CORA_FAIL(Error::wrong_value,
              std::string("cora: ") + what + " must be a numpy array or a torch tensor");
}

/// A vector (..., n) from Python as a column (..., n, 1): a torch tensor of any batch, or a
/// numpy array or list (1-D, or a batch (..., n)), or a number.
inline Tensor vector_of(const py::handle &o, const char *what) {
#ifdef CORACPP_PYTHON_TORCH
    py::detail::make_caster<torch::Tensor> torch_caster;
    if (torch_caster.load(o, /*convert=*/false)) return column(py::cast<torch::Tensor>(o));
#endif
    if (py::isinstance<py::array>(o) || py::isinstance<py::list>(o) ||
        py::isinstance<py::tuple>(o)) {
        py::array a = py::array_t<double, py::array::forcecast>::ensure(o);
        if (a && a.ndim() >= 1) return host_tensor(o, false).expand_dims(-1);
        if (a && a.ndim() == 0) return host_tensor(a.reshape({1}), true);
    }
    CORA_FAIL(Error::wrong_value,
              std::string("cora: ") + what +
                  " must be a vector: a torch tensor (..., n) or a numpy array");
}

// Sets --------------------------------------------------------------------------------------------

/// Any cora set as a Set: a copy of a library set, the object itself for a Python subclass.
inline Set set_of(const py::handle &o) {
    if (py::isinstance<Zonotope>(o)) return Set(py::cast<Zonotope>(o));
    if (py::isinstance<ConZonotope>(o)) return Set(py::cast<ConZonotope>(o));
    if (py::isinstance<Interval>(o)) return Set(py::cast<Interval>(o));
    if (py::isinstance<EmptySet>(o)) return Set(py::cast<EmptySet>(o));
    if (py::isinstance<Fullspace>(o)) return Set(py::cast<Fullspace>(o));
    if (py::isinstance<ContSet>(o)) {
        // the Python object owns the C++ part; the Set keeps the object alive
        const ContSet *S = py::cast<const ContSet *>(o);
        py::object keep = py::reinterpret_borrow<py::object>(o);
        auto release = [keep](const ContSet *) mutable {
            if (!python_alive()) {
                keep.release(); // the interpreter is gone: leak, as Python does at exit
                return;
            }
            py::gil_scoped_acquire gil;
            keep = py::object();
        };
        return Set(std::shared_ptr<const ContSet>(S, release));
    }
    CORA_FAIL(Error::wrong_value, "cora: expected a set (Zonotope, ConZonotope, Interval, "
                                  "EmptySet, Fullspace, a ContSet), got " +
                                      std::string(py::str(py::type::of(o).attr("__name__"))));
}

/// A Set as the Python object of its class.
inline py::object object_of(const Set &S) {
    switch (S.class_id()) {
    case ClassId::zonotope:
        return py::cast(S.as<Zonotope>());
    case ClassId::con_zonotope:
        return py::cast(S.as<ConZonotope>());
    case ClassId::interval:
        return py::cast(S.as<Interval>());
    case ClassId::empty_set:
        return py::cast(S.as<EmptySet>());
    case ClassId::fullspace:
        return py::cast(S.as<Fullspace>());
    default:
        return py::cast(&S.get(), py::return_value_policy::reference);
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
