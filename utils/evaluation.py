import time
import openai

from tqdm import tqdm
from typing import Dict, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

EVALUATION_SYSTEM_MESSAGE = (
    "You will be given a query and a reference text. "
    "You must determine whether the reference text contains an answer to the input query. "
    "Your response must be binary (0 or 1) and "
    "should not contain any text or characters aside from 0 or 1. "
    "0 means that the reference text does not contain an answer to the query. "
    "1 means the reference text contains an answer to the query."
)
QUERY_CONTEXT_PROMPT_TEMPLATE = """# Query: {query}

# Reference: {reference}

# Binary: """


# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def evaluate_query_and_retrieved_context(query: str, context: str, model_name: str) -> str:
    time.sleep(1)  # Solve hang issue
    prompt = QUERY_CONTEXT_PROMPT_TEMPLATE.format(
        query=query,
        reference=context,
    )
    response = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": EVALUATION_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        request_timeout=30
    )
    return response["choices"][0]["message"]["content"]


def evaluate_retrievals(
        retrievals_data: Dict[str, str],
        model_name: str,
) -> List[str]:
    responses = []
    for query, retrieved_context in tqdm(retrievals_data.items()):
        response = evaluate_query_and_retrieved_context(
            query, retrieved_context, model_name)
        responses.append(response)
    return responses


def process_binary_responses(
        binary_responses: List[str], binary_to_string_map: Dict[int, str]
) -> List[str]:
    """
    Parse binary responses and convert to the desired format
    converts them to the desired format. The binary_to_string_map parameter
    should be a dictionary mapping binary values (0 or 1) to the desired
    string values (e.g. "irrelevant" or "relevant").
    """
    processed_responses = []
    for binary_response in binary_responses:
        try:
            binary_value = int(binary_response.strip())
            processed_response = binary_to_string_map[binary_value]
        except (ValueError, KeyError):
            processed_response = None
        processed_responses.append(processed_response)
    return processed_responses
