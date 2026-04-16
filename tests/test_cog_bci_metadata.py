from pathlib import Path

from emotiv_learn.cog_bci_metadata import (
    build_nback_condition_labels,
    load_kss_scores,
    load_rsme_scores,
    NBackRecordingSummary,
    load_trigger_codes,
    summarize_nback_events,
)


def test_parse_cog_bci_questionnaire_files(tmp_path: Path) -> None:
    rsme_path = tmp_path / "RSME.txt"
    rsme_path.write_text(
        "\n".join(
            [
                "sbj,Session,Condition,Score,condition",
                "1,1,4,30,ZeroBack",
                "1,1,5,60,OneBack",
                "1,1,6,90,TwoBack",
                "1,1,7,80,PVT",
            ]
        ),
        encoding="utf-8",
    )
    kss_path = tmp_path / "KSS.txt"
    kss_path.write_text(
        "\n".join(
            [
                "sbj,sess,score,condition,Condition",
                "1,1,2,0,beginning",
                "1,1,6,1,end",
            ]
        ),
        encoding="utf-8",
    )

    labels = build_nback_condition_labels(
        rsme_scores=load_rsme_scores(rsme_path),
        kss_scores=load_kss_scores(kss_path),
    )

    assert [label.condition for label in labels] == ["ZeroBack", "OneBack", "TwoBack"]
    assert [label.difficulty_level for label in labels] == [0, 1, 2]
    assert labels[2].workload_estimate == 0.6
    assert labels[2].kss_beginning == 0.125
    assert labels[2].kss_end == 0.625


def test_parse_trigger_codes(tmp_path: Path) -> None:
    trigger_path = tmp_path / "triggerlist.txt"
    trigger_path.write_text("code,content\n600,ZEROBACK Start\n6032,ZEROBACK Correct Response\n", encoding="utf-8")

    triggers = load_trigger_codes(trigger_path)

    assert triggers[0].code == "600"
    assert triggers[1].content == "ZEROBACK Correct Response"


def test_summarize_nback_events_estimates_accuracy_and_rt() -> None:
    events = [
        {"type": "6022", "latency": 100.0},
        {"type": "6032", "latency": 175.0},
        {"type": "6023", "latency": 300.0},
        {"type": "6033", "latency": 450.0},
        {"type": "6021", "latency": 600.0},
    ]

    summary = summarize_nback_events(events=events, condition="ZeroBack", srate=100.0)

    assert summary["stimulus_count"] == 3
    assert summary["response_count"] == 2
    assert summary["response_accuracy"] == 0.5
    assert summary["mean_response_time_s"] == 1.125

def test_nback_recording_summary_dataclass_shape() -> None:
    summary = NBackRecordingSummary("sub-01", "1", "ZeroBack", 0, 0.2, 1.0, 0.4, 20, 60, 0.1, 0.2)
    assert summary.subject_id == "sub-01"
    assert summary.difficulty_level == 0
