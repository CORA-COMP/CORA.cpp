"""errors - the exceptions of CORA.cpp, one class per CORA identifier, and the warning category

Every error the library raises is a cora.CORAError; its class names CORA's identifier
(`cora.errors.DimensionMismatch` for CORA:dimensionMismatch) and also derives from the built-in
exception that fits, so `except ValueError` catches a wrong value as it would in numpy.

    try:
        Z + cora.Zonotope(np.zeros(3), np.eye(3))
    except cora.errors.DimensionMismatch as e:
        print(e.identifier, e)

Warnings use the category cora.CORAWarning; silence them with the warnings module.
"""


class CORAError(Exception):
    """An error of CORA.cpp; `identifier` is CORA's ("CORA:dimensionMismatch")."""

    identifier = "CORA:error"


class CORAWarning(UserWarning):
    """A warning of CORA.cpp."""


# The C++ name of each identifier (cora::Error::Id), CORA's spelling and the built-in base.
_IDENTIFIERS = [
    ("wrong_input_in_constructor", "CORA:wrongInputInConstructor", ValueError),
    ("no_input_in_set_constructor", "CORA:noInputInSetConstructor", TypeError),
    ("dimension_mismatch", "CORA:dimensionMismatch", ValueError),
    ("empty_set", "CORA:emptySet", ValueError),
    ("converter_issue", "CORA:converterIssue", ValueError),
    ("config_file", "CORA:configFile", OSError),
    ("file_not_found", "CORA:fileNotFound", FileNotFoundError),
    ("wrong_value", "CORA:wrongValue", ValueError),
    ("no_exact_alg", "CORA:noExactAlg", NotImplementedError),
    ("no_specific_alg", "CORA:noSpecificAlg", NotImplementedError),
    ("solver_issue", "CORA:solverIssue", RuntimeError),
    ("no_suitable_solver", "CORA:noSuitableSolver", RuntimeError),
    ("not_converged", "CORA:notConverged", RuntimeError),
    ("empty_property", "CORA:emptyProperty", ValueError),
    ("wrong_field_value", "CORA:wrongFieldValue", ValueError),
    ("plot_properties", "CORA:plotProperties", ValueError),
    ("even_number_input_args", "CORA:evenNumberInputArgs", TypeError),
    ("odd_number_input_args", "CORA:oddNumberInputArgs", TypeError),
    ("num_input_args_constructor", "CORA:numInputArgsConstructor", TypeError),
    ("redundant_name_value_pair", "CORA:redundantNameValuePair", TypeError),
    ("degenerate_set", "CORA:degenerateSet", ValueError),
    ("yalmip", "CORA:YALMIP", RuntimeError),
    ("out_of_domain", "CORA:outOfDomain", ValueError),
    ("special_error", "CORA:specialError", RuntimeError),
    ("not_supported", "CORA:notSupported", NotImplementedError),
    ("not_defined", "CORA:notDefined", ValueError),
    ("nn_layer_not_supported", "CORA:nnLayerNotSupported", NotImplementedError),
    ("reach_set_explosion", "CORA:reachSetExplosion", RuntimeError),
    ("out_of_memory", "CORA:outOfMemory", MemoryError),
    ("noops", "CORA:noops", TypeError),
    ("install", "CORA:install", ImportError),
    ("report_to_dev", "CORA:reportToDev", RuntimeError),
]


def _class_name(name):
    """dimension_mismatch -> DimensionMismatch."""
    return "".join(part.capitalize() for part in name.split("_"))


def _make(name, identifier, base):
    doc = f"CORA's {identifier}; also a {base.__name__}."
    return type(_class_name(name), (CORAError, base), {"identifier": identifier, "__doc__": doc})


BY_NAME = {name: _make(name, identifier, base) for name, identifier, base in _IDENTIFIERS}
globals().update({cls.__name__: cls for cls in BY_NAME.values()})

__all__ = ["CORAError", "CORAWarning"] + [cls.__name__ for cls in BY_NAME.values()]
