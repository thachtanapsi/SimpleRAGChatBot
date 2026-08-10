from rag_app.cli import build_parser


def test_reindex_command_is_available_without_arguments():
    args = build_parser().parse_args(["reindex"])

    assert args.command == "reindex"
