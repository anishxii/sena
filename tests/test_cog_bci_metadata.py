from pathlib import Path

from emotiv_learn.validation.cog_bci import (
    build_nback_condition_labels,
    load_kss_scores,
    load_rsme_scores,
    load_trigger_codes,
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
