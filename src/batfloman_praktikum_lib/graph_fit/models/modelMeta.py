from typing import cast, List, Type

class ModelMeta(type):
    def __add__(cls, other):
        from .compositeFitModel import CompositeFitModel, make_static_model_full

        # Collect components from both sides
        cls_components = getattr(cls, "_components", [cls])
        other_components = getattr(other, "_components", [other])
        new_components = cls_components + other_components

        model_func, get_param_names_func, get_initial_guess_func = make_static_model_full(new_components)

        class MergedComposite(CompositeFitModel):
            _components = new_components
            model = model_func
            get_param_names = get_param_names_func
            get_initial_guess = get_initial_guess_func

        return MergedComposite

    def __mul__(cls, other):
        # Wenn der linke Operand die Multiplikation ist, leiten wir an __rmul__ weiter
        return cls.__rmul__(other)

    def __rmul__(cls, other):
        if isinstance(other, int):
            from .compositeFitModel import CompositeFitModel, make_static_model_full
            from .fitModel import FitModel

            components = [cls] * other
            components = cast(List[Type[FitModel]], components)
            model_func, get_param_names_func, get_initial_guess_func = make_static_model_full(components)

            class RepeatedComposite(CompositeFitModel):
                _components = components
                model = model_func
                get_param_names = get_param_names_func
                get_initial_guess = get_initial_guess_func

            return RepeatedComposite
        else:
            return NotImplemented
