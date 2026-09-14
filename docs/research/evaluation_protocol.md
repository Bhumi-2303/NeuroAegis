# Evaluation Protocol

- **Window Metrics**: Evaluated at a strict 0.5 classification threshold.
- **Event Metrics**: A seizure event is detected if at least one window overlapping the event by $\ge$ 50% is flagged as positive.
- **False Alarms (FA/day)**: Merged contiguous false-positive windows into distinct episodes, normalized over 24 hours.
- **Detection Delay**: Measured from the true seizure onset to the onset of the first correctly classified overlapping window.
