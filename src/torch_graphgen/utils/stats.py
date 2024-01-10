PROPERTY_NAMES = ["out_channels", "out_features"]


def get_n_components(module):
    for property in PROPERTY_NAMES:
        if hasattr(module, property):
            return getattr(module, property)

    raise AttributeError(
        f"Module {module} is not currently supported - file an issue on Github to provide your point of view on how to implement this."
    )
