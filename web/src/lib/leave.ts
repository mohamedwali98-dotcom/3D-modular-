import type { Job } from '../api/types';

/** What Cancel on Analyzing does: a finished analysis is handed over (never thrown away because its playback had
 * not caught up), a running one is cancelled on the server (freeing its slot), a stopped one is just left. */
export function onLeave(job: Job | null): 'hand_over' | 'cancel' | 'leave' {
  if (job?.status === 'done' && job.result) return 'hand_over';
  if (!job || job.status === 'running') return 'cancel';
  return 'leave';
}
