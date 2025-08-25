# TODO: Remove the same thing from openai, anthropic, etc, and use this instead


def _with_tracer_wrapper(func):
    def _with_tracer(tracer, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer
