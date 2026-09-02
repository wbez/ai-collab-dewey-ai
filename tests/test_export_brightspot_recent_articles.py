import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "export_brightspot_recent_articles.py"
)
SPEC = importlib.util.spec_from_file_location("export_brightspot_recent_articles", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_item(
    item_id: str,
    comparison_text: str,
    publish_date_ms: int,
) -> dict:
    return {
        "id": item_id,
        "headline": comparison_text.splitlines()[0],
        "content": comparison_text,
        "url": f"https://example.com/{item_id}",
        "authors": ["Author"],
        "publish_date": "2026-01-01T00:00:00.000Z",
        "_comparison_text": comparison_text,
        "_comparison_md5": MODULE.md5_text(comparison_text),
        "_publish_date_ms": publish_date_ms,
    }


def test_exact_duplicates_are_removed_by_md5_before_similarity_checks() -> None:
    text = "Shared headline\n\nsame content"
    items = [
        make_item("older", text, 1),
        make_item("newer", text, 2),
    ]

    deduped, exact_removed, similar_removed = MODULE.dedupe_normalized_articles(items)

    assert [item["id"] for item in deduped] == ["older"]
    assert exact_removed == 1
    assert similar_removed == 0


def test_similar_duplicates_are_shortlisted_and_keep_latest_item() -> None:
    left = (
        "Test headline\n\n"
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    )
    right = (
        "Test headline\n\n"
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicrpn"
    )
    items = [
        make_item("older", left, 1),
        make_item("newer", right, 2),
    ]

    deduped, exact_removed, similar_removed = MODULE.dedupe_normalized_articles(items)

    assert [item["id"] for item in deduped] == ["newer"]
    assert exact_removed == 0
    assert similar_removed == 1


def test_different_articles_are_not_deduped() -> None:
    items = [
        make_item("one", "Headline one\n\nbody one", 1),
        make_item("two", "Headline two\n\nbody two with different text", 2),
    ]

    deduped, exact_removed, similar_removed = MODULE.dedupe_normalized_articles(items)

    assert [item["id"] for item in deduped] == ["one", "two"]
    assert exact_removed == 0
    assert similar_removed == 0
