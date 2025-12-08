from lightspeed_evaluation.core.models import EvaluationResult

from pydantic import Field
from typing import Optional


class EvaluationResultExt(EvaluationResult):
    judge_id: Optional[str] = Field(
        default=None,
        description="Judge identifier for panel of judges evaluations",
    )


if __name__ == '__main__':
    # Sanity check 
    print("Child fields:", EvaluationResultExt.model_fields)