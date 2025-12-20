from __future__ import annotations

import pytest


def test_external_manifest_row_sha_validation() -> None:
    from apps.api.ingest.external_sources_manifest import ExternalSourceManifestRow

    with pytest.raises(Exception):
        ExternalSourceManifestRow(
            source_id="SRC1",
            title="t",
            author="a",
            year=2020,
            license="ok",
            file_path="data/external_sources/x.txt",
            sha256="not-a-sha",
            file_format="txt",
            language="ar",
            weight=1.0,
        )


def test_sha256_file_bytes(tmp_path) -> None:
    from apps.api.ingest.external_sources_manifest import sha256_file_bytes

    p = tmp_path / "a.txt"
    p.write_bytes(b"abc")
    h = sha256_file_bytes(p)
    assert h == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

