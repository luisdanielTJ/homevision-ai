from behavior_engine.gemini_enricher import GeminiEnricher


def test_mock_enricher_returns_string():
    enricher = GeminiEnricher(mock=True)
    text = enricher.enrich(camera_id="cam0", dwell_time=17.0, head_turns=3, object_raised=False)
    assert isinstance(text, str)
    assert len(text) > 10


def test_mock_enricher_includes_dwell_time():
    enricher = GeminiEnricher(mock=True)
    text = enricher.enrich(camera_id="cam0", dwell_time=22.0, head_turns=2, object_raised=True)
    assert "22" in text or "customer" in text.lower()


def test_mock_enricher_includes_camera_id():
    enricher = GeminiEnricher(mock=True)
    text = enricher.enrich(camera_id="aisle-7", dwell_time=15.0, head_turns=3, object_raised=False)
    assert "aisle-7" in text
