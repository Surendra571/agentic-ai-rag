"""Unit tests for the groundedness verification evaluator."""

from unittest.mock import MagicMock
import pytest

from app.config import get_settings
from app.grading.groundedness import (
    GroundednessAPIError,
    GroundednessEvaluator,
    GroundednessResult,
    MissingAPIKeyError,
    check_groundedness,
    format_evaluator_prompt,
)


@pytest.fixture
def mock_chat_client() -> MagicMock:
    """Fixture providing a mocked chat client with structured output support."""
    client = MagicMock()
    client.model_name = "gpt-4o-mini"
    structured_invoker = MagicMock()
    client.with_structured_output.return_value = structured_invoker
    return client


def test_fully_grounded_answer(mock_chat_client: MagicMock) -> None:
    """1. Fully grounded answer: Context supports all factual claims."""
    mock_result = GroundednessResult(
        grounded=True,
        grounding_score=0.96,
        unsupported_claims=[],
    )
    mock_chat_client.with_structured_output.return_value.invoke.return_value = mock_result

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    query = "What is an autonomous loop?"
    draft_answer = "Autonomous loops continuously perceive context, plan actions, and execute tools."
    retrieved_context = [
        "Autonomous loops continuously perceive context, plan actions, and execute tools to achieve goals."
    ]

    result = evaluator.evaluate(
        query=query,
        draft_answer=draft_answer,
        retrieved_context=retrieved_context,
    )

    assert result.grounded is True
    assert result.grounding_score == 0.96
    assert result.unsupported_claims == []
    assert result.to_dict() == {
        "grounded": True,
        "grounding_score": 0.96,
        "unsupported_claims": [],
    }


def test_partially_unsupported_answer(mock_chat_client: MagicMock) -> None:
    """2. Partially unsupported answer: Some claims are absent from the context."""
    mock_result = GroundednessResult(
        grounded=False,
        grounding_score=0.65,
        unsupported_claims=["quantum teleportation processors"],
    )
    mock_chat_client.with_structured_output.return_value.invoke.return_value = mock_result

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    query = "What components make up an agent?"
    draft_answer = (
        "Agents comprise memory, planning, tools, and quantum teleportation processors."
    )
    retrieved_context = [
        "An agent architecture incorporates memory, planning algorithms, and tool integrations."
    ]

    result = evaluator.evaluate(
        query=query,
        draft_answer=draft_answer,
        retrieved_context=retrieved_context,
    )

    assert result.grounded is False
    assert result.grounding_score == 0.65
    assert "quantum teleportation processors" in result.unsupported_claims


def test_completely_unsupported_answer(mock_chat_client: MagicMock) -> None:
    """3. Completely unsupported answer: None of the claims are grounded in context."""
    mock_result = GroundednessResult(
        grounded=False,
        grounding_score=0.0,
        unsupported_claims=["Invented by ancient Roman philosophers in 44 BC"],
    )
    mock_chat_client.with_structured_output.return_value.invoke.return_value = mock_result

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    query = "When was Agentic AI created?"
    draft_answer = "Agentic AI was invented by ancient Roman philosophers in 44 BC."
    retrieved_context = [
        "Agentic AI frameworks emerged with modern large language models in the 2020s."
    ]

    result = evaluator.evaluate(
        query=query,
        draft_answer=draft_answer,
        retrieved_context=retrieved_context,
    )

    assert result.grounded is False
    assert result.grounding_score == 0.0
    assert len(result.unsupported_claims) == 1


def test_empty_context(mock_chat_client: MagicMock) -> None:
    """4. Empty context: Insufficient context results in grounded=False without calling LLM."""
    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    # Empty list
    result_empty = evaluator.evaluate(
        query="Explain tools",
        draft_answer="Agents call tools.",
        retrieved_context=[],
    )

    assert result_empty.grounded is False
    assert result_empty.grounding_score == 0.0
    assert len(result_empty.unsupported_claims) > 0
    mock_chat_client.with_structured_output.return_value.invoke.assert_not_called()

    # Context with only whitespace
    result_whitespace = evaluator.evaluate(
        query="Explain tools",
        draft_answer="Agents call tools.",
        retrieved_context=["   ", "\n\t"],
    )

    assert result_whitespace.grounded is False
    assert result_whitespace.grounding_score == 0.0
    mock_chat_client.with_structured_output.return_value.invoke.assert_not_called()


def test_unsupported_claims_strictly_forces_grounded_false(
    mock_chat_client: MagicMock,
) -> None:
    """Ensure grounding rule: if unsupported claims exist, grounded MUST be false even if LLM returned true."""
    # Defective LLM output claiming grounded=True with unsupported claims
    defect_result = GroundednessResult(
        grounded=True,
        grounding_score=1.0,
        unsupported_claims=["Invented a perpetual motion machine"],
    )
    mock_chat_client.with_structured_output.return_value.invoke.return_value = defect_result

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    result = evaluator.evaluate(
        query="What did the author build?",
        draft_answer="The author built a perpetual motion machine.",
        retrieved_context=["The author built an agent workflow."],
    )

    assert result.grounded is False
    assert result.grounding_score < 1.0


def test_groundedness_api_failure(mock_chat_client: MagicMock) -> None:
    """Test upstream LLM API errors raise GroundednessAPIError."""
    mock_chat_client.with_structured_output.return_value.invoke.side_effect = RuntimeError(
        "OpenAI service unavailable"
    )

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    with pytest.raises(GroundednessAPIError, match="Groundedness evaluation failed"):
        evaluator.evaluate(
            query="Explain agents",
            draft_answer="Agents think and act.",
            retrieved_context=["Agents think and act."],
        )


def test_missing_api_key_raises_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test initializing default evaluator without API key raises MissingAPIKeyError."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    with pytest.raises(MissingAPIKeyError, match="OpenAI API key is missing"):
        GroundednessEvaluator(api_key="")

    get_settings.cache_clear()


def test_check_groundedness_convenience_function(mock_chat_client: MagicMock) -> None:
    """Test convenience wrapper function check_groundedness."""
    mock_result = GroundednessResult(
        grounded=True,
        grounding_score=0.91,
        unsupported_claims=[],
    )
    mock_chat_client.with_structured_output.return_value.invoke.return_value = mock_result

    evaluator = GroundednessEvaluator(chat_client=mock_chat_client)

    res = check_groundedness(
        query="What is RAG?",
        draft_answer="RAG retrieves context before generation.",
        retrieved_context=["RAG retrieves context before generation."],
        evaluator=evaluator,
    )

    assert res.grounded is True
    assert res.grounding_score == 0.91


def test_format_evaluator_prompt() -> None:
    """Test prompt formatting isolates query, answer, and context chunks."""
    prompt = format_evaluator_prompt(
        query="What is tool use?",
        draft_answer="Tools give agents external actions.",
        retrieved_context=["Tools provide actions.", "Agents call external APIs."],
    )

    assert "USER QUERY:\nWhat is tool use?" in prompt
    assert "DRAFT ANSWER TO EVALUATE:\nTools give agents external actions." in prompt
    assert "RETRIEVED CONTEXT:" in prompt
    assert "[Context Chunk 1]\nTools provide actions." in prompt
    assert "[Context Chunk 2]\nAgents call external APIs." in prompt
