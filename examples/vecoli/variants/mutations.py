from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


def apply_variant(
    sim_data: "SimulationDataEcoli", params: dict[str, Any]
) -> "SimulationDataEcoli":
    """
    Apply attribute mutations to sim_data via dot-path traversal.

    Args:
        sim_data: Simulation data to modify
        params: Parameter dictionary of the following format::

            {
                "mutations": {
                    "path.to.attribute": value,
                    ...
                }
            }

        Indexed array mutations use the special form::

            {
                "path.to.array": {"__index__": idx, "__value__": val},
            }

    Returns:
        Simulation data with each ``path.to.attribute`` set to ``value``.
    """
    mutations = params.get("mutations", {})
    for attr_path, value in mutations.items():
        parts = attr_path.split(".")
        obj = sim_data
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if isinstance(value, dict) and "__index__" in value:
            arr = getattr(obj, parts[-1])
            arr[value["__index__"]] = value["__value__"]
        else:
            setattr(obj, parts[-1], value)

    return sim_data
