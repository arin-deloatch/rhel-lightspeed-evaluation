"""Extended API client that supports chat/completions endpoint."""

import logging
from typing import Any, Optional
from diskcache import Cache
import httpx

from lightspeed_evaluation.core.api import APIClient as BaseAPIClient
from lightspeed_evaluation.core.models import APIConfig
from lightspeed_evaluation.core.models import APIConfig, APIRequest, APIResponse
from rhel_lightspeed_evaluation.extensions.core.models.system import APIConfigExt
from lightspeed_evaluation.core.system.exceptions import APIError

from rhel_lightspeed_evaluation.extensions.core.models.api import APIRequestExt

logger = logging.getLogger(__name__)


class APIClientExt(BaseAPIClient):
    """Extended API client that supports 'chat/completions' endpoint type.
    
    This extends the base APIClient to handle the 'chat/completions' endpoint
    type in addition to the default 'streaming' and 'query' endpoints.
    
    For 'chat/completions', it maps to the base client's 'query' endpoint
    but constructs the URL as '/chat/completions' instead.
    """

    def __init__(self, config: APIConfig | APIConfigExt):
        """Initialize the extended API client.
        
        Args:
            api_config: API configuration (APIConfig or APIConfigExt)
        """
        self._is_chat_completions = config.endpoint_type == "chat/completions"
        config.endpoint_type = "query" if self._is_chat_completions else config.endpoint_type
        super().__init__(config)

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,) -> APIResponse:
        """Query the API endpoint.
        
        For 'chat/completions' endpoint, constructs the URL as:
        {api_base}/{version}/chat/completions
        
        For other endpoint types ('query' or 'streaming'), delegates to
        the base class implementation.
        
        Args:
            *args: Positional arguments passed to the query method
            **kwargs: Keyword arguments passed to the query method
            
        Returns:
            The response from the API endpoint
        """
        if self._is_chat_completions:
            return self._query_chat_completions(query, conversation_id, attachments)
        else:
            # For "streaming" or "query" endpoints, delegate to base implementation
            return super().query(query, conversation_id, attachments)
    
    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
    ) -> APIResponse:
        """Query the API using the configured endpoint type.

        Args:
            query: The question/query to ask
            conversation_id: Optional conversation ID for context
            attachments: Optional list of attachments

        Returns:
            APIResponse with Response, Tool calls, Conversation ID
        """
        if not self.client:
            raise APIError("API client not initialized")

        api_request = self._prepare_request(query, conversation_id, attachments)
        if self.config.cache_enabled:
            cached_response = self._get_cached_response(api_request)
            if cached_response is not None:
                logger.debug("Returning cached response for query: '%s'", query)
                return cached_response

        if self._is_chat_completions:
            response = self._chat_completions_query(api_request)
        elif self.endpoint_type == "streaming":
            response = self._streaming_query(api_request)
        else:
            response = self._standard_query(api_request)

        if self.config.cache_enabled:
            self._add_response_to_cache(api_request, response)

        return response
    
    def _prepare_request(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
    ) -> APIRequestExt:
        """Prepare API request with common parameters."""
        return APIRequestExt.create(
            query=query,
            messages=[{"role": "user", "content": query}],
            provider=self.config.provider,
            model=self.config.model,
            no_tools=self.config.no_tools,
            conversation_id=conversation_id,
            system_prompt=self.config.system_prompt,
            attachments=attachments,
        )

    def _chat_completions_query(self, api_request: APIRequest) -> APIResponse:
        """Query the API using chat/completions endpoint."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            response = self.client.post(
                f"/{self.version}/chat/completions",
                json=api_request.model_dump(exclude_none=True),
            )
            response.raise_for_status()

            response_data = response.json()["choices"][0]["message"]
            if "content" not in response_data:
                raise APIError("API response missing 'content' field")

            # Format tool calls to match streaming endpoint format
            # Currently only compatible with OLS
            if "tool_calls" in response_data and response_data["tool_calls"]:
                raw_tool_calls = response_data["tool_calls"]
                formatted_tool_calls = []

                # Convert list[dict] to list[list[dict]] format
                for tool_call in raw_tool_calls:
                    if isinstance(tool_call, dict):
                        formatted_tool = {
                            "tool_name": tool_call.get("tool_name")
                            or tool_call.get("name")  # Current OLS
                            or "",
                            "arguments": tool_call.get("arguments")
                            or tool_call.get("args")  # Current OLS
                            or {},
                        }
                        formatted_tool_calls.append([formatted_tool])

                response_data["tool_calls"] = formatted_tool_calls

            response_data["response"] = response_data["content"]
            response_data["conversation_id"] = response.json()["id"]
            return APIResponse.from_raw_response(response_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("standard", self.timeout) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except ValueError as e:
            raise self._handle_validation_error(e) from e
        except APIError:
            raise
        except Exception as e:
            raise self._handle_unexpected_error(e, "standard query") from e
