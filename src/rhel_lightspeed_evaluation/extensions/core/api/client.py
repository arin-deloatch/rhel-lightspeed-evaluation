"""Extended API client that supports chat/completions endpoint."""

import logging
from copy import deepcopy
from typing import Any

import httpx
from lightspeed_evaluation.core.api import APIClient as BaseAPIClient
from lightspeed_evaluation.core.models import APIConfig, APIRequest
from lightspeed_evaluation.core.system.exceptions import APIError

from rhel_lightspeed_evaluation.extensions.core.models.api import APIRequestExt, APIResponseExt
from rhel_lightspeed_evaluation.extensions.core.models.system import APIConfigExt

logger = logging.getLogger(__name__)


def _format_tool_calls(raw_tool_calls: list[Any]) -> list[list[dict[str, Any]]]:
    """Normalize raw tool call dicts into list[list[dict]] format expected by APIResponse."""
    formatted: list[list[dict[str, Any]]] = []
    for tool_call in raw_tool_calls:
        if isinstance(tool_call, dict):
            formatted.append(
                [
                    {
                        "tool_name": tool_call.get("tool_name") or tool_call.get("name") or "",
                        "arguments": tool_call.get("arguments") or tool_call.get("args") or {},
                    }
                ]
            )
    return formatted


class APIClientExt(BaseAPIClient):
    """Extended API client that supports 'chat/completions' endpoint type.

    For chat/completions, overrides the base query() to use _chat_completions_query.
    For all other endpoint types (infer, streaming, query), delegates to the base
    class which already handles routing.
    """

    def __init__(self, config: APIConfig | APIConfigExt):
        normalized_config = deepcopy(config)
        self._is_chat_completions = normalized_config.endpoint_type == "chat/completions"
        if self._is_chat_completions:
            normalized_config.endpoint_type = "query"
        super().__init__(normalized_config)

    def query(
        self,
        query: str,
        conversation_id: str | None = None,
        attachments: list[str] | None = None,
        extra_request_params: dict[str, Any] | None = None,
    ) -> APIResponseExt:
        """Query the API using the configured endpoint type."""
        if not self._is_chat_completions:
            return super().query(query, conversation_id, attachments, extra_request_params)

        if not self.client:
            raise APIError("API client not initialized")

        api_request = self._prepare_request(
            query, conversation_id, attachments, extra_request_params
        )
        if self.config.cache_enabled:
            cached_response = self._get_cached_response(api_request)
            if cached_response is not None:
                logger.debug("Returning cached response for query: '%s'", query)
                return cached_response

        response = self._chat_completions_query(api_request)

        if self.config.cache_enabled:
            self._add_response_to_cache(api_request, response)

        return response

    def _prepare_request(
        self,
        query: str,
        conversation_id: str | None = None,
        attachments: list[str] | None = None,
        extra_request_params: dict[str, Any] | None = None,
    ) -> APIRequestExt:
        """Prepare API request with common parameters."""
        resolved_extra = {**(self.config.extra_request_params or {})}
        if extra_request_params:
            resolved_extra.update(extra_request_params)
        return APIRequestExt.create(
            query=query,
            messages=[{"role": "user", "content": query}],
            provider=self.config.provider,
            model=self.config.model,
            no_tools=self.config.no_tools,
            conversation_id=conversation_id,
            system_prompt=self.config.system_prompt,
            attachments=attachments,
            extra_request_params=resolved_extra or None,
        )

    def _chat_completions_query(self, api_request: APIRequest) -> APIResponseExt:
        """Query the API using chat/completions endpoint."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            response = self.client.post(
                f"/{self.config.version}/chat/completions",
                json=self._serialize_request(api_request),
            )
            response.raise_for_status()

            full_response = response.json()
            response_data = full_response["choices"][0]["message"]
            if "content" not in response_data:
                raise APIError("API response missing 'content' field")

            if "tool_calls" in response_data and response_data["tool_calls"]:
                response_data["tool_calls"] = _format_tool_calls(response_data["tool_calls"])

            response_data["response"] = response_data["content"]
            response_data["conversation_id"] = full_response["id"]

            if "rag_chunks" in full_response:
                response_data["rag_chunks"] = full_response["rag_chunks"]

            usage = full_response.get("usage", {})
            if usage:
                response_data.setdefault("input_tokens", usage.get("prompt_tokens", 0))
                response_data.setdefault("output_tokens", usage.get("completion_tokens", 0))

            return APIResponseExt.from_raw_response(response_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("chat/completions", self.config.timeout) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except ValueError as e:
            raise self._handle_validation_error(e) from e
        except APIError:
            raise
        except Exception as e:
            raise self._handle_unexpected_error(e, "chat/completions query") from e
