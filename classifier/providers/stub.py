import json

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

# Deterministic constants — realistic in shape, obviously fake in content so they are
# never mistaken for real extraction results.
_STUB_DIAGNOSIS_RESPONSE = json.dumps(
    {
        "reasoning": "STUB_REASONING — deterministic stub response, no model invoked.",
        "diagnoses": [
            {"name": "STUB_DIAGNOSIS", "icd_code": "Z00.0"},
        ],
    }
)

STUB_TREATMENT_REQUEST_RESPONSE = json.dumps(
    {
        "treatment_requested": False,
        "reasoning": "STUB_REASONING — deterministic stub response, no model invoked.",
    }
)


STUB_MAIN_DIAGNOSIS_RESPONSE = json.dumps(
    {
        "main_diagnosis": "STUB DIAGNOSIS",
        "diagnosis_category": "Other",
        "icd_code": "Z00.0",
        "reasoning": "STUB_REASONING — deterministic stub response, no model invoked.",
    }
)


STUB_PATIENT_AGE_RESPONSE = json.dumps(
    {
        "age_years": 99.0,
        "dob": "01/01/1900",
        "date_of_service": "01/01/1999",
        "reasoning": "STUB_REASONING — deterministic stub response, no model invoked.",
    }
)


def create_stub_chat_model(response: str | None = None) -> BaseChatModel:
    """Return a fake chat model that always emits a fixed, deterministic response.

    Used for laptop dev with sensitive data: no real model invoked, nothing leaves
    the machine. Defaults to a constant diagnosis-extraction payload.
    """
    return FakeListChatModel(responses=[response or _STUB_DIAGNOSIS_RESPONSE])
