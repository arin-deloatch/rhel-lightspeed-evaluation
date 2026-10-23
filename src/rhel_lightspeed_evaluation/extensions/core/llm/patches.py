"""Install RHEL judge integrations once in each evaluation process."""

_applied = False


def apply_patches() -> None:
    """Apply before importing the pipeline, including in spawned workers."""
    global _applied
    if _applied:
        return

    from rhel_lightspeed_evaluation.extensions.core.llm.deepeval_json_patch import (
        apply_deepeval_json_patch,
    )
    from rhel_lightspeed_evaluation.extensions.core.llm.deepeval_patch import (
        apply_deepeval_watsonx_patch,
    )
    from rhel_lightspeed_evaluation.extensions.core.llm.vertex_dedicated_patch import (
        apply_vertex_dedicated_patch,
    )
    from rhel_lightspeed_evaluation.extensions.core.llm.vertex_params_patch import (
        apply_vertex_params_patch,
    )

    apply_deepeval_watsonx_patch()
    apply_vertex_params_patch()
    apply_vertex_dedicated_patch()
    apply_deepeval_json_patch()
    _applied = True
