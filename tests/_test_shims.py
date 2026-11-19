"""Lightweight test shims for environments where FastAPI/Starlette are not installed."""
from __future__ import annotations
import sys
import types


def install_fastapi_shims():
    try:
        import fastapi  # noqa: F401
        import starlette.datastructures  # noqa: F401
        return
    except ImportError:
        pass

    fastapi_mod = types.ModuleType("fastapi")
    responses_mod = types.ModuleType("fastapi.responses")
    staticfiles_mod = types.ModuleType("fastapi.staticfiles")
    pydantic_mod = types.ModuleType("pydantic")
    starlette_mod = types.ModuleType("starlette")
    datastructures_mod = types.ModuleType("starlette.datastructures")

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail=None):
            super().__init__(f"{status_code}: {detail}")
            self.status_code = status_code
            self.detail = detail

    class UploadFile:
        def __init__(self, filename: str = "upload.txt", file=None):
            self.filename = filename
            self.file = file

    def File(default=None):
        return default

    def Form(default=None):
        return default

    class FastAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.routes = []

        def mount(self, *args, **kwargs):
            pass

        def middleware(self, *args, **kwargs):
            def wrap(fn):
                return fn
            return wrap

        def _decorator(self, method, path):
            def wrap(fn):
                self.routes.append((method, path, fn))
                return fn
            return wrap

        def get(self, path, **kwargs):
            return self._decorator("GET", path)

        def post(self, path, **kwargs):
            return self._decorator("POST", path)

        def put(self, path, **kwargs):
            return self._decorator("PUT", path)

        def patch(self, path, **kwargs):
            return self._decorator("PATCH", path)

        def delete(self, path, **kwargs):
            return self._decorator("DELETE", path)

    class FileResponse:
        def __init__(self, path, media_type=None, filename=None):
            self.path = path
            self.media_type = media_type
            self.filename = filename

    class JSONResponse:
        def __init__(self, content, status_code=200):
            self.content = content
            self.status_code = status_code

    class StaticFiles:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FieldSpec:
        def __init__(self, default=None, default_factory=None, **kwargs):
            self.default = default
            self.default_factory = default_factory
            self.kwargs = kwargs

    def Field(default=None, default_factory=None, **kwargs):
        return _FieldSpec(default=default, default_factory=default_factory, **kwargs)

    class BaseModel:
        def __init__(self, **kwargs):
            for cls in reversed(self.__class__.__mro__):
                for k, v in getattr(cls, "__dict__", {}).items():
                    if k.startswith("_") or callable(v):
                        continue
                    if isinstance(v, _FieldSpec):
                        if v.default_factory is not None:
                            setattr(self, k, v.default_factory())
                        else:
                            setattr(self, k, v.default)
                    else:
                        setattr(self, k, v)
            for k, v in kwargs.items():
                setattr(self, k, v)

    fastapi_mod.FastAPI = FastAPI
    fastapi_mod.UploadFile = UploadFile
    fastapi_mod.File = File
    fastapi_mod.Form = Form
    fastapi_mod.HTTPException = HTTPException
    responses_mod.FileResponse = FileResponse
    responses_mod.JSONResponse = JSONResponse
    staticfiles_mod.StaticFiles = StaticFiles
    pydantic_mod.BaseModel = BaseModel
    pydantic_mod.Field = Field
    datastructures_mod.UploadFile = UploadFile

    sys.modules.setdefault("fastapi", fastapi_mod)
    sys.modules.setdefault("fastapi.responses", responses_mod)
    sys.modules.setdefault("fastapi.staticfiles", staticfiles_mod)
    sys.modules.setdefault("pydantic", pydantic_mod)
    sys.modules.setdefault("starlette", starlette_mod)
    sys.modules.setdefault("starlette.datastructures", datastructures_mod)
