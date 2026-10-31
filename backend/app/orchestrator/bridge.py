"""Backend -> simulator bridge over the simulator's loopback HTTP API."""

import httpx


class SimUnavailable(Exception):
    pass


class SimBridge:
    def __init__(self, base_url: str = "http://127.0.0.1:8001", client: httpx.Client | None = None,
                 timeout_s: float = 3.0):
        self.base_url = base_url
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout_s)

    def _request(self, method: str, path: str, **kw) -> httpx.Response:
        try:
            return self.client.request(method, path, **kw)
        except httpx.HTTPError as exc:
            raise SimUnavailable(f"simulator not reachable at {self.base_url}: {type(exc).__name__}") from exc

    def status(self) -> dict:
        r = self._request("GET", "/status")
        r.raise_for_status()
        return r.json()

    def scene(self) -> dict:
        r = self._request("GET", "/scene")
        r.raise_for_status()
        return r.json()

    def pick(self, request: dict) -> tuple[int, dict]:
        r = self._request("POST", "/pick", json=request)
        return r.status_code, r.json()

    def reset(self) -> dict:
        r = self._request("POST", "/reset")
        r.raise_for_status()
        return r.json()

    def cancel(self) -> None:
        self._request("POST", "/cancel")
