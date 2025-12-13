from apps.api.retrieve.arabic_morph import expand_query_terms, generate_token_variants


def test_generate_token_variants_strips_common_prefixes():
    """
    Expected: 'والصبر' produces variant 'الصبر' and 'صبر' (article stripped).
    Edge: variants stay >= 3 chars.
    """
    vars_ = generate_token_variants("والصبر")
    assert "الصبر" in vars_
    assert "صبر" in vars_


def test_expand_query_terms_includes_inflected_forms():
    terms = expand_query_terms("والمحاسبة والمساءلة")
    # Normalizer may reduce ة->ه depending on config; accept both.
    assert "محاسبة" in terms or "محاسبه" in terms


