"""Extended API client that supports chat/completions endpoint."""

import logging
from typing import Any

import httpx
from lightspeed_evaluation.core.api import APIClient as BaseAPIClient
from lightspeed_evaluation.core.models import APIConfig, APIRequest
from lightspeed_evaluation.core.system.exceptions import APIError

from rhel_lightspeed_evaluation.extensions.core.models.api import APIRequestExt, APIResponseExt
from rhel_lightspeed_evaluation.extensions.core.models.system import APIConfigExt

logger = logging.getLogger(__name__)

RAG_CHUNK_DELIMITER = "\n\n---\n\n"


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


def _extract_rag_chunks_from_tool_results(
    tool_results: list[Any],
) -> list[dict[str, str]]:
    """Extract RAG chunks from infer endpoint tool_results.

    MCP call results contain RAG content as a single string with chunks
    separated by '\\n\\n---\\n\\n'. Each delimited segment becomes a
    separate RAG chunk.
    """
    rag_chunks: list[dict[str, str]] = []
    for item in tool_results:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "mcp_call":
            continue
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            logger.debug("Skipping mcp_call tool_result with empty content")
            continue
        for chunk in content.split(RAG_CHUNK_DELIMITER):
            chunk = chunk.strip()
            if chunk:
                rag_chunks.append({"content": chunk})
    if not rag_chunks and tool_results:
        logger.warning(
            "tool_results present (%d items) but no mcp_call entries with RAG content found",
            len(tool_results),
        )
    return rag_chunks


class APIClientExt(BaseAPIClient):
    """Extended API client that supports 'chat/completions' endpoint type."""

    def __init__(self, config: APIConfig | APIConfigExt):
        self._is_chat_completions = config.endpoint_type == "chat/completions"
        self._is_infer = config.endpoint_type == "infer"
        if self._is_chat_completions or self._is_infer:
            config.endpoint_type = "query"
        super().__init__(config)

    def query(
        self,
        query: str,
        conversation_id: str | None = None,
        attachments: list[str] | None = None,
        extra_request_params: dict[str, Any] | None = None,
    ) -> APIResponseExt:
        """Query the API using the configured endpoint type."""
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

        if self._is_chat_completions:
            response = self._chat_completions_query(api_request)
        elif self._is_infer:
            response = self._infer_query(api_request)
        elif self.config.endpoint_type == "streaming":
            response = self._streaming_query(api_request)
        else:
            response = self._standard_query(api_request)

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

    def _serialize_infer_request(self, api_request: APIRequest) -> dict[str, Any]:
        """Serialize request for the infer endpoint (expects 'question', not 'query')."""
        payload: dict[str, Any] = {"question": api_request.query}
        extra = {**(api_request.extra_request_params or {})}
        for key, value in extra.items():
            if key not in payload:
                payload[key] = value
        return payload

    def _infer_query(self, api_request: APIRequest) -> APIResponseExt:
        """Query the API using the infer endpoint."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            response = self.client.post(
                f"/{self.config.version}/infer",
                json=self._serialize_infer_request(api_request),
            )
            response.raise_for_status()

            raw = response.json()
            data = raw.get("data", {})
            if not data or "text" not in data:
                raise APIError("API response missing 'data.text' field")

            response_data: dict[str, Any] = {
                "response": data["text"],
                "conversation_id": data.get("request_id", raw.get("request_id", "")),
            }

            if data.get("rag_chunks"):
                response_data["rag_chunks"] = data["rag_chunks"]
            elif data.get("tool_results"):
                response_data["rag_chunks"] = _extract_rag_chunks_from_tool_results(
                    data["tool_results"]
                )
            if "input_tokens" in data:
                response_data["input_tokens"] = data["input_tokens"]
            if "output_tokens" in data:
                response_data["output_tokens"] = data["output_tokens"]

            if data.get("tool_calls"):
                response_data["tool_calls"] = _format_tool_calls(data["tool_calls"])

            return APIResponseExt.from_raw_response(response_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("infer", self.config.timeout) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except ValueError as e:
            raise self._handle_validation_error(e) from e
        except APIError:
            raise
        except Exception as e:
            raise self._handle_unexpected_error(e, "infer query") from e

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
