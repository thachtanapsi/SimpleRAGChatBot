from rag_app.cli import build_parser


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
