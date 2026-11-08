# Online A/C phase: what the existing evidence can and cannot answer

This note reuses an existing event-time audit run; it is **not** the 120-step
capture in `common_state.txt`. The event-time numbers below describe a
different C run with 129 KFs and must not be numerically combined with the
120-step replay.

## Direct event-time evidence (not shutdown slices)

`/tmp/psl-room1-control/event-audit-room1-v2/event_quality/loop_evt1/summary.csv`
was evaluated at the LOOP event on 24 fixed views with zero optimizer steps:

| state | PSNR |
|---|---:|
| before camera + before model | 29.41824476 |
| after camera + before model | 27.36089857 |
| after camera + corrected model | 29.01224224 |
| Δ camera-only | −2.057346185 |
| Δ corrected map vs before | −0.4060025215 |

The same event's GBA measurement is:

```text
before 27.34581852 -> camera-only 27.64163725 -> corrected 27.3373332
```

So **for run C, the LOOP event contains a real instantaneous camera/map jump**
(-2.06 dB from the pose update; the map correction recovers 1.65 dB, leaving
-0.406 dB at the event). The GBA event is small and not a monotonic degradation.

The fixed-geometry event replay (`event-audit-room1-v2/common_abc`) starts at
the same 29.4182 and reaches 31.547 at 100 updates; that is an **event-time
recovery experiment**, not an online trajectory. It does not show a gradual
post-event decline in this controlled setting.

## What is not established

- There is no A-side LOOP event in the matched A/C pair. A has no event; C has
  a LOOP at KF 124/125. Therefore the A/C gap cannot be assigned to a common
  event.
- The A/C shutdown files provide 17 common `img_name` frames and an A−C GS gap
  of +1.0327 dB, but those scores are rendered at shutdown. The `frame001300`
  split is an image-time label, not evidence that quality was already lower
  before the event. It cannot answer “pre-LOOP vs event jump”.
- Existing logs contain KF/frame positions and the final optimizer counts, but
  not per-frame optimizer-step counts, SH/resolution switches, queue backlog,
  stop reason, or model checkpoints at event/frame boundaries. No historical
  quality claim is made from them.

## First answer

For the only event-bearing run (C): the event itself produces a measured jump
at LOOP; the map-corrected state still has a -0.406 dB event-time residual, and
the GBA event is small. A pre-LOOP formation of the A/C gap is **not measured**,
and a post-LOOP gradual growth is **not supported by the event-time replay**.
The A/C phase comparison is therefore unresolved without a same-event state
checkpoint; it must not be inferred from shutdown PSNR.

## Required next diagnostic (not started here)

Use one bounded online A/C diagnostic run with checkpoints at a few data-frame
indices and at any LOOP/GBA consumption. For each checkpoint record: actual
optimizer-step count, active SH/resolution, KF usage counts, queue/backlog and
stop state, plus the rendered native GS PSNR from a separate no-grad pass.
Also record the step at which each LOOP/GBA is consumed and the camera-change
range. Those states must be actual model snapshots; shutdown renders sliced by
image time are not acceptable as history.

Until that exists, the strict 120-update replay supports only “no sustained C
degradation under this error-guided replay”; it does not identify the online
phase where the A/C gap arises.
