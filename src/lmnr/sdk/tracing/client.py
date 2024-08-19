from .tracing_types import Span, Trace

from typing import Any, Union
import json
import logging
import requests


class APIError(Exception):
    def __init__(self, status: Union[int, str], message: str, details: Any = None):
        self.message = message
        self.status = status
        self.details = details

    def __str__(self):
        msg = "{0} ({1}): {2}"
        return msg.format(self.message, self.status, self.details)


class LaminarClient:
    _base_url = "https://api.lmnr.ai"

    def __init__(self, project_api_key: str):
        self.project_api_key = project_api_key

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.project_api_key,
            "Content-Type": "application/json",
        }

    def batch_post(self, batch: list[Union[Span, Trace]]):
        log = logging.getLogger("laminar.client")
        url = self._base_url + "/v2/traces"
        data = json.dumps({"traces": [item.to_dict() for item in batch]})
        log.debug(f"making request to {url}")
        headers = self._headers()
        res = requests.post(url, data=data, headers=headers)

        if res.status_code == 200:
            log.debug("data uploaded successfully")

        return self._process_response(
            res, success_message="data uploaded successfully", return_json=False
        )

    def _process_response(
        self, res: requests.Response, success_message: str, return_json: bool = True
    ) -> Union[requests.Response, Any]:
        log = logging.getLogger("laminar.client")
        log.debug("received response: %s", res.text)
        if res.status_code in (200, 201):
            log.debug(success_message)
            if return_json:
                try:
                    return res.json()
                except json.JSONDecodeError:
                    log.error("Response is not valid JSON.")
                    raise APIError(res.status_code, "Invalid JSON response received")
            else:
                return res
        try:
            payload = res.json()
            log.error("received error response: %s", payload)
            raise APIError(res.status_code, payload)
        except (KeyError, ValueError):
            raise APIError(res.status_code, res.text)
