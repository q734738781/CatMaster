import pytest
pytest.skip("integration smoke test; run manually", allow_module_level=True)

from openai import OpenAI

client = OpenAI()

# Minimal, no streaming, no reasoning summaries
resp = client.responses.create(
    model="gpt-5.2",
    input="ping",
    # stream defaults to false in Responses API
)

print(resp.output_text)
