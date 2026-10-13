"""Sum uninterrupted batches; omit a batch suspended for manuscript work."""
import json


def active_review_seconds(output,reviewer,decision,phenotype):
    path=output/'reviews'/reviewer/decision/(phenotype+'.json')
    if not path.exists():return float('nan')
    record=json.loads(path.read_text());timings=record.get('batch_seconds',[])
    total=sum(timings)
    pause_path=output/'report_pause.json'
    if pause_path.exists():
        pause=json.loads(pause_path.read_text())
        index=pause.get('next_batch_index')
        if (not pause.get('active') and pause.get('checkpoint')==str(path.relative_to(output))
                and index is not None and index<len(timings)):
            # The audit start was written after SIGSTOP, so its recorded
            # interval cannot recover the exact suspension duration. Omit
            # the entire affected batch instead of estimating its compute time.
            total-=timings[index]
    return total
