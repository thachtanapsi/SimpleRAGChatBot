from rag_app.cli import build_parser, main


def test_reindex_command_is_available_without_arguments():
    args = build_parser().parse_args(["reindex"])

    assert args.command == "reindex"


def test_serve_defaults_to_loopback():
    args = build_parser().parse_args(["serve"])

    assert args.host == "127.0.0.1"
    assert args.port == 8000


def test_serve_allows_explicit_container_bind():
    args = build_parser().parse_args(["serve", "--host", "0.0.0.0"])

    assert args.host == "0.0.0.0"


def test_full_report_backfill_modes_are_explicit():
    dry_run = build_parser().parse_args(["backfill-full-reports", "--dry-run"])
    apply = build_parser().parse_args(["backfill-full-reports", "--apply"])

    assert dry_run.command == "backfill-full-reports"
    assert dry_run.dry_run is True
    assert dry_run.apply is False
    assert apply.apply is True


def test_graph_reindex_modes_and_document_scope_are_explicit():
    dry_run = build_parser().parse_args(["graph-reindex", "--dry-run"])
    apply = build_parser().parse_args(
        ["graph-reindex", "--apply", "--document-id", "doc_123"]
    )

    assert dry_run.command == "graph-reindex"
    assert dry_run.dry_run is True
    assert dry_run.document_id is None
    assert apply.apply is True
    assert apply.document_id == "doc_123"


def test_advanced_eval_release_gate_is_explicit_and_legacy_eval_is_unchanged():
    legacy = build_parser().parse_args(["evaluate", "evals/questions.jsonl"])
    advanced = build_parser().parse_args(
        [
            "evaluate",
            "evals/advanced.jsonl",
            "--release-gate",
            "--validate-only",
        ]
    )

    assert legacy.release_gate is False
    assert legacy.validate_only is False
    assert advanced.release_gate is True
    assert advanced.validate_only is True

    scored = build_parser().parse_args(
        [
            "evaluate",
            "advanced.jsonl",
            "--release-gate",
            "--results",
            "results.jsonl",
            "--legacy-results",
            "current.json",
            "--legacy-baseline",
            "evals/hybrid_full.json",
        ]
    )
    assert scored.legacy_baseline == "evals/hybrid_full.json"


def test_advanced_eval_validation_error_is_clean_json(capsys, tmp_path):
    missing = tmp_path / "missing.jsonl"

    exit_code = main(
        ["evaluate", str(missing), "--release-gate", "--validate-only"]
    )

    assert exit_code == 2
    assert '"valid": false' in capsys.readouterr().err
