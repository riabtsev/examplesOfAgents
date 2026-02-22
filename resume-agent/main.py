import json
import os

import anthropic
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("resume-screener")

_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

SYSTEM_PROMPT = """You are an expert HR recruiter and technical interviewer.
Your task is to evaluate a candidate's resume against a job description and return a structured JSON assessment.
Respond ONLY with valid JSON in exactly this format:
{
  "score": <integer 1-10>,
  "summary": "<brief overall assessment>",
  "pros": ["<strength 1>", "<strength 2>", ...],
  "cons": ["<weakness 1>", "<weakness 2>", ...],
  "interview_questions": ["<question 1>", "<question 2>", ...]
}
"""


@mcp.tool()
async def screen_resume(job_description: str, resume_text: str) -> dict:
    """
    Screen a candidate's resume against a job description.

    Args:
        job_description: The job posting / requirements.
        resume_text: The candidate's resume in plain text.

    Returns:
        A dict with score, summary, pros, cons, and interview_questions.
    """
    user_message = (
        f"JOB DESCRIPTION:\n{job_description}\n\n"
        f"RESUME:\n{resume_text}"
    )

    message = _client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    if not message.content:
        raise ValueError("Empty response from Claude")

    raw = message.content[0].text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude returned non-JSON response: {raw!r}") from exc


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080, path="/mcp")
