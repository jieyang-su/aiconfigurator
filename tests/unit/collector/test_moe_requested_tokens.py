from collector.sglang import collect_moe


def test_explicit_token_filter_adds_shape_outside_model_grid(monkeypatch):
    monkeypatch.setattr(collect_moe, "get_sm_version", lambda: 90)
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    monkeypatch.setenv("COLLECTOR_MOE_TYPES", "fp8_block")
    monkeypatch.setenv("COLLECTOR_MOE_TP_SIZES", "1")
    monkeypatch.setenv("COLLECTOR_MOE_EP_SIZES", "8")
    monkeypatch.setenv("COLLECTOR_MOE_TOKENS", "24")
    monkeypatch.setenv("COLLECTOR_MOE_DISTRIBUTIONS", "balanced")

    cases = collect_moe.get_moe_test_cases()

    assert len(cases) == 1
    assert cases[0][1] == 24
