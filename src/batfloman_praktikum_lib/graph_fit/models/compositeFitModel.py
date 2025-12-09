from .fitModel import FitModel
from .modelMeta import ModelMeta

class CompositeFitModel(FitModel, metaclass=ModelMeta):
    @staticmethod
    def model(x, **params):
        raise NotImplementedError("Subclasses must implement model(x, *params)")

    @staticmethod
    def get_param_names():
        raise NotImplementedError("Subclasses must implement get_param_names()")

    @staticmethod
    def get_initial_guess(x, y):
        raise NotImplementedError("Subclasses must implement get_initial_guess()")

def make_static_model_full(components):
    # Build explicit parameter names
    param_names = []
    for idx, comp in enumerate(components):
        names = comp.get_param_names()
        for name in names:
            param_names.append(f"{name}_{idx+1}")

    # Build the model function
    args_str = "x, " + ", ".join(param_names)
    code_model = f"def model({args_str}):\n"
    code_model += "    result = 0\n"
    
    i = 0
    for comp_idx, comp in enumerate(components):
        n = len(comp.get_param_names())
        comp_params = param_names[i:i+n]
        comp_args = ", ".join(comp_params)
        code_model += f"    result += {comp.__name__}.model(x, {comp_args})\n"
        i += n
    code_model += "    return result\n"

    # Build get_param_names function
    code_params = "def get_param_names():\n"
    code_params += f"    return {param_names}\n"

    # Dynamic get_initial_guess method (requires x and y)
    def get_initial_guess(x, y):
        guesses = []
        for comp in components:
            guesses.extend(comp.get_initial_guess(x, y))
        return guesses

    # Execute model and param_names in namespace
    namespace = {comp.__name__: comp for comp in components}
    exec(code_model, namespace)
    exec(code_params, namespace)

    return (
        staticmethod(namespace["model"]),
        staticmethod(namespace["get_param_names"]),
        staticmethod(get_initial_guess)
    )
