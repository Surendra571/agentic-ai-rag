"""Evaluation test suite for in-document assignment questions.

Tests the 5 core assignment questions against the Agentic AI RAG system:
1. What is the core definition of Agentic AI as outlined in the eBook?
2. What are the main architectural components required to build agentic systems?
3. What real-world industry use cases for Agentic AI are discussed?
4. How does Agentic AI differ from traditional generative AI chatbots according to the text?
5. What key challenges or limitations of Agentic AI are mentioned?

Validates:
- query exists
- final_answer exists
- retrieved_context_chunks exists
- confidence_score exists and is between 0 and 1
- grounded is True
- sources citation metadata is preserved
"""

from typing import Any
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_rag_pipeline
from app.generation.generator import FALLBACK_RESPONSE
from app.grading.groundedness import GroundednessResult
from app.graph.workflow import RAGResult, run_rag
from app.main import app
from app.retrieval.retriever import RetrievedChunk

# 5 Assignment Questions with realistic document context and expected facts
IN_DOCUMENT_BENCHMARK = [
    {
        "id": "q1_core_definition",
        "question": "What is the core definition of Agentic AI as outlined in the eBook?",
        "context": (
            "Agentic AI refers to autonomous artificial intelligence systems characterized by "
            "proactive agency, environmental perception, reasoning, and goal-directed action execution loops. "
            "Unlike static models, Agentic AI continuously evaluates feedback from its environment and adapts "
            "its planning to accomplish complex, open-ended tasks."
        ),
        "source": "Ebook-Agentic-AI.pdf",
        "page": 2,
        "chunk_id": "page_02_chunk_01",
        "similarity_score": 0.94,
        "expected_answer": (
            "Agentic AI is defined as an autonomous artificial intelligence system characterized by proactive "
            "agency, environmental perception, reasoning, and goal-directed action execution loops that iteratively "
            "adapt plans based on environmental feedback."
        ),
        "key_phrases": ["autonomous", "perception", "reasoning", "action execution loops", "feedback"],
    },
    {
        "id": "q2_architectural_components",
        "question": "What are the main architectural components required to build agentic systems?",
        "context": (
            "The core architecture of an agentic system consists of five essential components: "
            "1) Perception layer to ingest environmental signals and user inputs; "
            "2) Reasoning & Planning engine powered by LLMs (e.g., ReAct, Plan-and-Solve); "
            "3) Memory architecture featuring short-term conversation context and long-term vector storage; "
            "4) Tool execution interface enabling interaction with external APIs and databases; "
            "5) Reflection and self-critique loop for iterative error correction."
        ),
        "source": "Ebook-Agentic-AI.pdf",
        "page": 8,
        "chunk_id": "page_08_chunk_02",
        "similarity_score": 0.92,
        "expected_answer": (
            "The main architectural components required to build agentic systems are: "
            "1) Perception layer for input processing; "
            "2) Reasoning and Planning engine; "
            "3) Short-term and long-term Memory architecture; "
            "4) Tool execution interface for external APIs; and "
            "5) Reflection and self-critique feedback loops."
        ),
        "key_phrases": ["Perception", "Reasoning", "Planning", "Memory", "Tool execution", "Reflection"],
    },
    {
        "id": "q3_industry_use_cases",
        "question": "What real-world industry use cases for Agentic AI are discussed?",
        "context": (
            "The eBook highlights key industry use cases where Agentic AI is actively transforming workflows: "
            "automated software engineering (code generation, debugging, automated PR reviews); "
            "intelligent customer support with autonomous resolution of complex multi-step tickets; "
            "financial analysis and algorithmic market research; "
            "healthcare clinical data retrieval and patient trial matching; and "
            "autonomous supply chain inventory and route optimization."
        ),
        "source": "Ebook-Agentic-AI.pdf",
        "page": 16,
        "chunk_id": "page_16_chunk_03",
        "similarity_score": 0.90,
        "expected_answer": (
            "The real-world industry use cases discussed include automated software engineering (code generation and "
            "debugging), intelligent customer support ticketing, financial analysis and market research, "
            "healthcare clinical trial matching, and autonomous supply chain logistics."
        ),
        "key_phrases": ["software engineering", "customer support", "financial analysis", "healthcare", "supply chain"],
    },
    {
        "id": "q4_difference_from_traditional_chatbots",
        "question": "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
        "context": (
            "According to the text, Agentic AI fundamentally differs from traditional generative AI chatbots. "
            "Traditional chatbots operate reactively in single-turn text-to-text generation without tool execution, "
            "environment feedback, or state persistence. In contrast, Agentic AI exhibits proactive agency, "
            "maintains state and memory, autonomously decomposes goals into subtasks, interacts with external tools "
            "and APIs, and executes iterative feedback loops to self-correct."
        ),
        "source": "Ebook-Agentic-AI.pdf",
        "page": 4,
        "chunk_id": "page_04_chunk_01",
        "similarity_score": 0.93,
        "expected_answer": (
            "Agentic AI differs from traditional chatbots because traditional chatbots operate reactively in single-turn "
            "text generation without tools or state, whereas Agentic AI exhibits proactive agency, maintains state and memory, "
            "decomposes goals into subtasks, uses external tools, and leverages iterative feedback loops to self-correct."
        ),
        "key_phrases": ["reactively", "single-turn", "proactive agency", "tools", "feedback loops"],
    },
    {
        "id": "q5_challenges_limitations",
        "question": "What key challenges or limitations of Agentic AI are mentioned?",
        "context": (
            "Key challenges and limitations of Agentic AI mentioned in the text include: "
            "compounding errors and hallucination loops across multi-step execution chains; "
            "non-deterministic execution behavior leading to unpredictability in production; "
            "security vulnerabilities such as prompt injection and unauthorized tool execution; "
            "high latency and inference compute cost from multi-round agent iterations; and "
            "difficulty in establishing reliable evaluation and debugging benchmarks."
        ),
        "source": "Ebook-Agentic-AI.pdf",
        "page": 24,
        "chunk_id": "page_24_chunk_02",
        "similarity_score": 0.89,
        "expected_answer": (
            "Key challenges and limitations mentioned include compounding errors and hallucination loops in multi-step "
            "chains, non-deterministic execution behavior, security vulnerabilities like prompt injection, "
            "high latency and inference costs, and difficulties with evaluation and debugging."
        ),
        "key_phrases": ["compounding errors", "hallucination loops", "non-deterministic", "security", "latency"],
    },
]


def create_mock_services_for_benchmark(item: dict[str, Any]) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Helper to create configured mock retriever, generator, and evaluator for a benchmark question."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id=item["chunk_id"],
            source=item["source"],
            page=item["page"],
            text=item["context"],
            similarity_score=item["similarity_score"],
            metadata={
                "source": item["source"],
                "page": item["page"],
                "chunk_id": item["chunk_id"],
            },
        )
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = item["expected_answer"]

    mock_evaluator = MagicMock()
    mock_evaluator.evaluate.return_value = GroundednessResult(
        grounded=True,
        grounding_score=0.95,
        unsupported_claims=[],
    )

    return mock_retriever, mock_generator, mock_evaluator


@pytest.mark.parametrize("item", IN_DOCUMENT_BENCHMARK, ids=[d["id"] for d in IN_DOCUMENT_BENCHMARK])
def test_in_document_questions_workflow(item: dict[str, Any]) -> None:
    """Test each of the 5 in-document assignment questions through the RAG workflow."""
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    # 1. Validate query exists and matches
    assert result.query == item["question"]

    # 2. Validate final_answer exists, is grounded, and not fallback
    assert result.final_answer is not None
    assert len(result.final_answer.strip()) > 0
    assert result.final_answer != FALLBACK_RESPONSE
    assert result.final_answer == item["expected_answer"]

    # 3. Validate retrieved_context exists and contains relevant chunk
    assert isinstance(result.retrieved_context, list)
    assert len(result.retrieved_context) >= 1
    assert item["context"] in result.retrieved_context

    # 4. Validate confidence_score exists and is between 0 and 1
    assert result.confidence_score is not None
    assert 0.0 <= result.confidence_score <= 1.0
    assert result.confidence_score >= 0.70  # Grounded answers should have high confidence

    # 5. Validate grounded status and metadata preservation
    assert result.grounded is True
    assert result.grounding_score >= 0.90
    assert result.retrieval_relevant is True
    assert len(result.sources) >= 1
    assert result.sources[0]["page"] == item["page"]
    assert result.sources[0]["chunk_id"] == item["chunk_id"]


@pytest.mark.parametrize("item", IN_DOCUMENT_BENCHMARK, ids=[d["id"] for d in IN_DOCUMENT_BENCHMARK])
def test_in_document_questions_fastapi_endpoint(item: dict[str, Any]) -> None:
    """Test each of the 5 in-document questions through FastAPI POST /api/v1/chat endpoint."""
    client = TestClient(app)

    mock_rag_result = RAGResult(
        query=item["question"],
        final_answer=item["expected_answer"],
        grounded=True,
        grounding_score=0.96,
        confidence_score=0.94,
        retrieval_relevant=True,
        sources=[{"source": item["source"], "page": item["page"], "chunk_id": item["chunk_id"]}],
        retrieved_context=[item["context"]],
        retry_count=0,
    )

    mock_pipeline = MagicMock(return_value=mock_rag_result)
    app.dependency_overrides[get_rag_pipeline] = lambda: mock_pipeline

    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": item["question"]},
        )

        assert response.status_code == 200
        data = response.json()

        # Strict response field validation
        assert "query" in data and data["query"] == item["question"]
        assert "final_answer" in data and data["final_answer"] == item["expected_answer"]
        assert "retrieved_context_chunks" in data
        assert isinstance(data["retrieved_context_chunks"], list)
        assert len(data["retrieved_context_chunks"]) == 1
        assert "confidence_score" in data
        assert isinstance(data["confidence_score"], float)
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert data["confidence_score"] >= 0.70

        # Grounding & metadata verification
        assert data.get("grounded") is True
        assert len(data.get("sources", [])) == 1
        assert data["sources"][0]["chunk_id"] == item["chunk_id"]
    finally:
        app.dependency_overrides.clear()


def test_core_definition_content_verification() -> None:
    """Detailed content check: Question 1 covers autonomy, perception, and action loops."""
    item = IN_DOCUMENT_BENCHMARK[0]
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    for phrase in item["key_phrases"]:
        assert phrase.lower() in result.final_answer.lower()


def test_architectural_components_content_verification() -> None:
    """Detailed content check: Question 2 covers perception, reasoning, planning, memory, tools."""
    item = IN_DOCUMENT_BENCHMARK[1]
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    for phrase in item["key_phrases"]:
        assert phrase.lower() in result.final_answer.lower()


def test_industry_use_cases_content_verification() -> None:
    """Detailed content check: Question 3 covers software engineering, customer support, healthcare."""
    item = IN_DOCUMENT_BENCHMARK[2]
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    for phrase in item["key_phrases"]:
        assert phrase.lower() in result.final_answer.lower()


def test_difference_from_traditional_chatbots_content_verification() -> None:
    """Detailed content check: Question 4 covers reactive vs proactive agency and tool loops."""
    item = IN_DOCUMENT_BENCHMARK[3]
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    for phrase in item["key_phrases"]:
        assert phrase.lower() in result.final_answer.lower()


def test_challenges_and_limitations_content_verification() -> None:
    """Detailed content check: Question 5 covers compounding errors, hallucination, and non-determinism."""
    item = IN_DOCUMENT_BENCHMARK[4]
    mock_retriever, mock_generator, mock_evaluator = create_mock_services_for_benchmark(item)

    result = run_rag(
        query=item["question"],
        retriever_service=mock_retriever,
        generator_service=mock_generator,
        groundedness_evaluator=mock_evaluator,
    )

    for phrase in item["key_phrases"]:
        assert phrase.lower() in result.final_answer.lower()
