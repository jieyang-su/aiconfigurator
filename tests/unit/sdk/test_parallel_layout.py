from aiconfigurator_core.sdk.system_spec import ParallelLayout


def _layout(**kwargs):
    return ParallelLayout(policy="tp_first", **kwargs)


def test_independent_policy_preserves_size_only_locality():
    layout = ParallelLayout(tp=8, pp=2, dp=2, cp=1, policy="independent")
    assert layout.group_is_local("pp", 8, 2)
    assert not layout.group_is_local("tp_dp", 8, 16)


def test_independent_pipeline_preserves_historical_pipeline_bandwidth():
    layout = ParallelLayout(tp=8, pp=2, policy="independent")
    spec = {"node": {"num_gpus_per_node": 8, "intra_node_bw": 100.0, "inter_node_bw": 10.0}}
    assert layout.bandwidth(spec, "pp", 2) == 10.0


def test_single_supernode_p2p_uses_intra_node_for_both_policies():
    spec = {
        "node": {
            "topology_scope": "single_supernode",
            "num_gpus_per_node": 8,
            "intra_node_bw": 100.0,
            "inter_node_bw": 0.0,
        }
    }
    for policy in ("independent", "tp_first"):
        layout = ParallelLayout(tp=4, pp=2, policy=policy)
        assert layout.bandwidth(spec, "pp", 2) == 100.0


def test_tp_first_keeps_tp_local_but_marks_pipeline_crossing():
    layout = _layout(tp=8, pp=2, dp=1, cp=1)
    assert layout.group_is_local("tp", 8, 8)
    assert not layout.group_is_local("pp", 8, 2)


def test_tp_first_pipeline_is_local_when_two_stages_fit():
    layout = _layout(tp=4, pp=2, dp=1, cp=1)
    assert layout.group_is_local("pp", 8, 2)


def test_tp_first_pipeline_checks_every_replicated_lane():
    layout = _layout(tp=2, pp=2, dp=2, cp=1)
    assert not layout.group_is_local("pp", 2, 2)


def test_tp_first_distinguishes_cp_and_full_attention_group():
    layout = _layout(tp=4, pp=1, dp=1, cp=4)
    assert layout.group_is_local("tp", 8, 4)
    assert not layout.group_is_local("cp", 8, 4)
    assert not layout.group_is_local("attention", 8, 16)


def test_tp_first_models_moe_tp_ep_as_full_worker_group():
    layout = _layout(tp=4, pp=1, dp=1, cp=1, moe_tp=2, moe_ep=2)
    assert layout.group_is_local("moe_tp", 8, 2)
    assert layout.group_is_local("moe_ep", 8, 2)
    assert layout.group_is_local("moe_tp_ep", 8, 4)


def test_tp_first_group_crosses_node_when_attention_width_exceeds_node():
    layout = _layout(tp=8, pp=1, dp=2, cp=1)
    assert not layout.group_is_local("attention", 8, 16)
