"""Rule extractor implementation assembly.

Reason: keep public `rule_extractor.py` small while splitting logic into mixins.
"""

from __future__ import annotations

from apps.api.ingest.rule_extractor_impl_build_finalize import _RuleExtractorBuildFinalizeMixin
from apps.api.ingest.rule_extractor_impl_extract import _RuleExtractorInitExtractMixin
from apps.api.ingest.rule_extractor_impl_headings import _RuleExtractorHeadingsMixin
from apps.api.ingest.rule_extractor_impl_process import _RuleExtractorProcessMixin
from apps.api.ingest.rule_extractor_impl_sections import _RuleExtractorSectionsMixin


class RuleExtractor(
    _RuleExtractorInitExtractMixin,
    _RuleExtractorProcessMixin,
    _RuleExtractorHeadingsMixin,
    _RuleExtractorSectionsMixin,
    _RuleExtractorBuildFinalizeMixin,
):
    """Concrete rule extractor class (assembled from mixins)."""

    pass

