"""Common constants for evaluation framework."""

from lightspeed_evaluation.core import constants

SUPPORTED_CSV_COLUMNS = [
    "conversation_group_id",
    "turn_id",
    "metric_identifier",
    "judge_id",
    "result",
    "score",
    "threshold",
    "reason",
    "execution_time",
    "query",
    "response",
]

if __name__ == '__main__':
    print(constants.DEFAULT_API_BASE)
    print(SUPPORTED_CSV_COLUMNS)