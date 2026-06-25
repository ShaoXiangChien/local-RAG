from rag_demo.chat.schemas import InferenceMetrics
from rag_demo.ui.chat_page import _performance_detail_rows, _performance_metric_rows


def test_performance_metric_rows_format_core_generation_metrics() -> None:
    metrics = InferenceMetrics(
        time_to_first_token_ms=321,
        generation_elapsed_ms=2500,
        prompt_eval_count=120,
        eval_count=25,
        prompt_tokens_per_second=200.0,
        output_tokens_per_second=20.0,
        estimated_effective_gflops=120.0,
    )

    rows = _performance_metric_rows(metrics)

    assert rows == [
        {"Metric": "TTFT", "Value": "321 ms"},
        {"Metric": "Output throughput", "Value": "20.00 tok/s"},
        {"Metric": "Prompt eval throughput", "Value": "200.00 tok/s"},
        {"Metric": "Output tokens", "Value": "25"},
        {"Metric": "Generation time", "Value": "2500 ms"},
        {"Metric": "Estimated effective throughput", "Value": "120.00 GFLOP/s"},
    ]


def test_performance_detail_rows_label_estimates_and_raw_ollama_timings() -> None:
    metrics = InferenceMetrics(
        model_name="llama3.2:3b",
        estimated_prompt_tokens=620,
        included_source_count=4,
        answer_chunk_count=2,
        thinking_chunk_count=0,
        answer_character_count=15,
        total_turn_elapsed_ms=3400,
        total_duration_ns=2_500_000_000,
        load_duration_ns=100_000_000,
        prompt_eval_count=120,
        prompt_eval_duration_ns=600_000_000,
        eval_count=25,
        eval_duration_ns=1_250_000_000,
        estimated_model_parameter_count=3_000_000_000,
    )

    rows = _performance_detail_rows(metrics)

    assert {"Metric": "Model", "Value": "llama3.2:3b"} in rows
    assert {"Metric": "Total turn time", "Value": "3400 ms"} in rows
    assert {"Metric": "Ollama total duration", "Value": "2500 ms"} in rows
    assert {"Metric": "Ollama load duration", "Value": "100 ms"} in rows
    assert {"Metric": "Prompt tokens (app estimate)", "Value": "620"} in rows
    assert {"Metric": "Prompt tokens (Ollama)", "Value": "120"} in rows
    assert {"Metric": "Eval duration", "Value": "1250 ms"} in rows
    assert {"Metric": "Estimated model parameters", "Value": "3.00B"} in rows
    assert {
        "Metric": "FLOP/s formula",
        "Value": "output tok/s * 2 * estimated params",
    } in rows
