from apps.api.retrieve.entity_resolver import EntityResolver


def test_entity_resolver_matches_morph_variants():
    resolver = EntityResolver()
    resolver.load_entities(
        pillars=[],
        core_values=[],
        sub_values=[{"id": "SV001", "name_ar": "الصبر"}],
    )

    # Query uses prefixed/plural-ish form
    results = resolver.resolve("كيف نحقق والصبر في حياتنا؟")
    assert any(r.entity_id == "SV001" and r.match_type in ("exact", "normalized", "morph") for r in results)


def test_entity_resolver_does_not_overmatch_short_common_words():
    resolver = EntityResolver()
    resolver.load_entities(
        pillars=[],
        core_values=[],
        sub_values=[{"id": "SV001", "name_ar": "الحق"}],
    )

    # "حقي" should not automatically match "الحق" at high confidence
    results = resolver.resolve("حقي في العمل")
    assert all(r.confidence < 0.9 for r in results)


