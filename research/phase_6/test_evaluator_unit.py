import mne
import sys
sys.path.insert(0, "/Volumes/BLACK-BOX/NeuroAegis")

from research.phase_6.siena_zero_shot_evaluator import SienaZeroShotEvaluator

evaluator = SienaZeroShotEvaluator()
raw = mne.io.read_raw_edf('/Volumes/BLACK-BOX/NeuroAegis/data/siena_edf/PN00/PN00-4.edf', preload=True, verbose=False)
res = evaluator.evaluate_recording(
    recording_id='PN00/PN00-4.edf',
    raw_eeg_data=raw.get_data(),
    channel_names=raw.ch_names
)

print(f"Evaluated PN00/PN00-4.edf:")
print(f"  Windows: {res['n_windows']}, Hours: {res['duration_hours']:.2f}")
print(f"  Inference time: {res['inference_time_sec']:.2f}s")
print(f"  Merged alarms count: {len(res['merged_alarms'])}")
for al in res['merged_alarms']:
    print(f"    Alarm: [{al['start_sec']:.1f}s - {al['end_sec']:.1f}s], Max Prob: {al['max_prob']:.4f}")
print(f"  Event evaluations: {res['event_evals']}")
print(f"  False alarms count: {res['n_false_alarms']}, FA/24h: {res['fa_per_24h']:.2f}")
