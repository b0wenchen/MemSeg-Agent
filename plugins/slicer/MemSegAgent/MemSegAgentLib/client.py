"""Loopback-only HTTP client; remote GPUs are reached through SSH forwarding."""

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

MAX_PIXELS = 2048 * 2048
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_ARCHIVE_BYTES = 192 * 1024 * 1024


def pack_array(array):
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError("Use a uint8 grayscale or RGB slice")
    if min(array.shape[:2]) < 2 or np.prod(array.shape[:2]) > MAX_PIXELS:
        raise ValueError("Slice exceeds the backend pixel limit")
    if array.ndim == 3 and array.shape[2] != 3:
        raise ValueError("RGB slices require three channels")
    return {
        "dtype": "uint8",
        "shape": list(array.shape),
        "data": base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii"),
    }


def unpack_mask(payload, shape):
    if payload.get("dtype") != "uint8" or payload.get("shape") != list(shape):
        raise ValueError("Backend mask dimensions do not match the selected slice")
    size = int(np.prod(shape))
    data = payload.get("data")
    if not isinstance(data, str) or len(data) != 4 * ((size + 2) // 3):
        raise ValueError("Invalid backend mask size")
    raw = base64.b64decode(data, validate=True)
    if len(raw) != size:
        raise ValueError("Invalid backend mask length")
    result = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
    if not np.isin(result, [0, 1]).all():
        raise ValueError("Backend returned a non-binary mask")
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Backend redirects are disabled")


class MemSegClient:
    def __init__(self, endpoint="http://127.0.0.1:8766", timeout=180):
        url = urllib.parse.urlsplit(endpoint)
        if (
            url.scheme != "http"
            or url.hostname not in ("localhost", "127.0.0.1", "::1")
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/")
        ):
            raise ValueError("Use a loopback HTTP address; forward remote servers with SSH")
        if url.port is not None and not 1024 <= url.port <= 65535:
            raise ValueError("Use a port between 1024 and 65535")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        # Patient images must not follow redirects or an environment HTTP proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, action, body=None, *, raw=False, payload=None):
        headers = {"X-MemSeg-Request": "1"}
        if payload is not None:
            data = payload
            headers["Content-Type"] = "application/zip"
        elif body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        else:
            data = None
        request = urllib.request.Request(
            self.endpoint + "/api/" + action, data=data, headers=headers
        )
        limit = MAX_ARCHIVE_BYTES if raw else MAX_RESPONSE_BYTES
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                result = response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read(4096)).get("detail", str(exc.code))
            except (ValueError, AttributeError):
                detail = str(exc.code)
            raise RuntimeError(f"MemSeg backend: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                "Cannot reach the MemSeg backend; check the server and forwarded port"
            ) from exc
        if len(result) > limit:
            raise ValueError("Backend response exceeds the size limit")
        return result if raw else json.loads(result)

    def connect(self):
        capabilities = self.request("capabilities")
        if capabilities.get("protocol") != "memseg-slicer-v1":
            raise ValueError("Update the MemSeg backend: incompatible Slicer API")
        return self.request("state")

    def predict(self, image):
        result = self.request("slice-predict", {"image": pack_array(image)})
        return unpack_mask(result["mask"], image.shape[:2]), result["retrieved"]

    def add_support(self, image, mask, name):
        return self.request(
            "slice-support", {"image": pack_array(image), "mask": pack_array(mask), "name": name}
        )

    def add_correction(self, image, mask, name, *, confirmed=False):
        if confirmed is not True:
            raise ValueError("Review and confirm the slice before saving it")
        return self.request(
            "slice-correct",
            {"image": pack_array(image), "mask": pack_array(mask), "name": name, "confirmed": True},
        )

    def export_archive(self):
        return self.request("export", raw=True)

    def import_archive(self, payload):
        if len(payload) > MAX_ARCHIVE_BYTES:
            raise ValueError("Archive exceeds 192 MiB")
        return self.request("import", payload=payload)
