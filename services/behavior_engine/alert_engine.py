class AlertEngine:
    def __init__(self, dwell_threshold: float, head_turn_threshold: int) -> None:
        self._dwell_threshold = dwell_threshold
        self._head_turn_threshold = head_turn_threshold

    def should_alert(self, dwell_time: float, head_turns: int, object_raised: bool) -> bool:
        dwell_met = dwell_time > self._dwell_threshold
        signal_met = head_turns > self._head_turn_threshold or object_raised
        return dwell_met and signal_met
