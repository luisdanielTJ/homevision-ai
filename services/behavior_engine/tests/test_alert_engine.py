from behavior_engine.alert_engine import AlertEngine


def test_no_alert_below_dwell_threshold():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=10.0, head_turns=3, object_raised=False) is False


def test_alert_when_dwell_met_and_head_turns_exceed_threshold():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=16.0, head_turns=3, object_raised=False) is True


def test_alert_when_dwell_met_and_object_raised():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=16.0, head_turns=0, object_raised=True) is True


def test_no_alert_when_dwell_met_but_no_signals():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=20.0, head_turns=0, object_raised=False) is False


def test_head_turn_threshold_is_exclusive():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=16.0, head_turns=2, object_raised=False) is False
    assert engine.should_alert(dwell_time=16.0, head_turns=3, object_raised=False) is True


def test_both_signals_still_requires_dwell():
    engine = AlertEngine(dwell_threshold=15.0, head_turn_threshold=2)
    assert engine.should_alert(dwell_time=5.0, head_turns=5, object_raised=True) is False
