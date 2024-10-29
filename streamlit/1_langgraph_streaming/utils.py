import inspect
from typing import Callable, TypeVar

from langchain_core.callbacks.base import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx


def get_streamlit_cb(parent_container: DeltaGenerator) -> BaseCallbackHandler:
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container: DeltaGenerator, initial_text: str = ""):
            self.container = container
            self.token_placeholder = self.container.empty()
            self.text = initial_text

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.token_placeholder.write(self.text)

    fn_return_type = TypeVar("fn_return_type")

    def add_streamlit_context(
        fn: Callable[..., fn_return_type]
    ) -> Callable[..., fn_return_type]:
        ctx = get_script_run_ctx()

        def wrapper(*args, **kwargs) -> fn_return_type:
            add_script_run_ctx(ctx=ctx)
            return fn(*args, **kwargs)

        return wrapper

    st_cb = StreamHandler(parent_container)

    for method_name, method_func in inspect.getmembers(
        st_cb, predicate=inspect.ismethod
    ):
        if method_name.startswith("on_"):
            setattr(st_cb, method_name, add_streamlit_context(method_func))

    return st_cb
