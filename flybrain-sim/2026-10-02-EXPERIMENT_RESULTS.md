# Fly-Brain Experiment Results — 2026-10-02

Session run by SeafoamBot research worker, 5:25am–1:00pm EDT.
All experiments on `cuda-kernel` branch, arena 1600×1200.

---

## 1. Gain Sweep (FLOW_GAIN)

**Method:** Swept FLOW_GAIN from 10→500, recorded DNa02 L/R spike rates per window.

**Results:**
- Gains 10, 20: DNa02 silent (0 spikes/window)
- Gain 50+: DNa02 saturates at ~5.5 spikes/window, completely flat through gain=500
- L/R differential **shrinks** with higher gain — both sides saturate equally

**Interpretation:** The FLOW_GAIN knob above ~50 is decorative. T4a input is already saturating DNa02 at threshold; increasing gain just drives both sides harder equally. The L/R difference needed for steering is *not* being amplified — it's being erased by bilateral saturation. `FLOW_GAIN=150` (current default) is well into the dead zone.

---

## 2. Ablation: T4a → DNa02 pathway (L/R encoding check)

**Method:** 300-frame run. Measured Pearson r(T4a_diff, DNa02_diff) and cross-correlations at lags 0–5.

**Results:**
- r(T4a_diff, DNa02_diff) = -0.0005 (across 300 frames)
- Cross-correlation at all lags 0–5: all < 0.02

**Interpretation:** T4a L/R differential has near-zero predictive power over DNa02 L/R output. DNa02 fires reliably (~5.5/side) but the L/R *difference* is stochastic noise, not encoded directional signal. The optic flow pathway is not steering the fly.

---

## 3. LPLC2 / DNp01 pathway check

**Method:** Same 300-frame ablation CSV. Computed Pearson r for:
- lplc2_diff vs min_dist_to_wall
- dnp01_diff vs lplc2_diff
- dnp01_l vs lplc2_l (ipsilateral)

**Results:**
| Correlation | r | p |
|---|---|---|
| lplc2_diff vs min_dist_wall | -0.011 | 0.85 |
| dnp01_diff vs lplc2_diff | 0.073 | 0.21 |
| lplc2_sum vs min_dist_wall | -0.015 | 0.80 |
| dnp01_l vs lplc2_l | 0.035 | 0.55 |
| dnp01_r vs lplc2_r | 0.021 | 0.71 |

Cross-correlation lplc2_diff → dnp01_diff at lags 0–5: all |r| < 0.08, no consistent pattern.

**Confound:** The fly stayed ~555px from all walls throughout all 300 frames (std=2.7px). Looming was never meaningfully triggered — LPLC2 firing at flat baseline (~1.95 left, ~1.68 right). The baseline asymmetry (L > R by 0.27) appears structural, not stimulus-driven.

**Interpretation:** LPLC2 → DNp01 pathway shows no directional encoding either, but the test is inconclusive because the fly never got close to a wall. A valid test requires deliberately driving the fly toward walls, or reducing the arena to force near-wall encounters.

---

## 4. Reseed Experiment (CPU active-set management)

**Method:** Ran 200 frames on CPU engine with and without per-frame `_reseed_driven()` call before each advance.

**Results:**
| Condition | DNa02_l mean | DNa02_r mean | nactive range |
|---|---|---|---|
| WITH per-frame reseed | 0.020 | 0.040 | 5366–123,753 |
| WITHOUT reseed (accumulates) | 0.000 | 0.000 | 8327–12,538 |

**Key findings:**
1. **Without reseed:** DNa02 = 0.000 exactly. Network converges to ~12k active neurons where inhibitory interneurons fully suppress DN output. Confirms the docstring warning.
2. **With reseed:** DNa02 = 0.020 — fires, but 275× less than CUDA's 5.5/side.
3. **CUDA vs CPU discrepancy:** The reseed question is CPU-only. CUDA engine ignores `nactive[]` entirely. The live simulation runs CUDA, so reseed has zero effect on the actual running system.
4. **nactive stability:** Without reseed, the network does NOT blow up to all 166k neurons as feared — it stabilizes at ~12k. Possibly a fixed-point attractor in the inhibitory subnetwork.

**Interpretation:** All prior gain_sweep and ablation results were CUDA runs. The CPU engine is a different (much weaker) dynamical system. Do not use CPU results to calibrate CUDA behavior.

---

## 5. DNp02 / DNp11 Trace (Task 5)

**Method:** `python3 trace_dn.py DNp02 DNp11 --top 10` against the full 166k-neuron connectome graph.

**LC4 downstream context (from separate LC4 trace):**
LC4 top downstream targets by weight: DNp04 L (1873), DNp04 R (1316), DNp01 L (1040), PVLP024 (×3, ~600–730), DNp01 R (710), **DNp02 L (627)**, AMMC-A1 (×2, ~580–616)

DNp02 is rank 6 in LC4 downstream (weight 627, bilateral). DNp11 has LC4 as only rank 8 upstream (weight 27.2) — a weak connection.

**DNp02 (2 neurons, bilateral):**

*Upstream top 10:* CB1280, CB2664, SAD053/055/064 — all central brain interneurons. No T4a, no LC4 in top 10. 1008 total upstream partners.

*Downstream top 10:* GNG004 (midline ganglia, weight 105), **pIP1** (weight 92), DNg108 L+R (weights 72–80), AN19B001 (×3, ~48–67), AN18B004 (×2, ~45–50), DNge119

**DNp11 (2 neurons, bilateral):**

*Upstream top 10:* SAD049/053/064, LC4 L (rank 8, weight 27.2), DNp01 L (rank 10, weight 27.0). 2823 total upstream partners.

*Downstream top 10:* AN19B001 (×2), DNg108 L+R, IN18B044 (×2), DNg97 L+R, GNG004, IN06B035

**Interpretation:**

1. **DNp02 is not strongly LC4-driven** — its top upstream partners are central brain interneurons (CB/SAD types), not visual pathway neurons. LC4 contributes ~627 weight but doesn't dominate DNp02's input.

2. **DNp11 has minimal LC4 connection** — LC4 appears at rank 8 with weight 27.2, essentially a background input.

3. **Shared downstream targets: DNg108 and GNG004** — both DNp02 and DNp11 converge on DNg108 (a descending neuron group not currently monitored in the sim) and GNG004 (gnathal ganglia midline). This suggests a convergent motor output pathway.

4. **pIP1 surprise** — pIP1 is the well-characterized male courtship chasing neuron. Its appearance as the #2 downstream target of DNp02 is unexpected in a looming/collision context. Either DNp02 has roles beyond looming avoidance, or pIP1 has broader functions than documented.

5. **DNg97 and DNge119** — additional descending neurons in DNp11's downstream that are not in the current sim's monitored group set. These are completely invisible to the current tracking.

6. **None of these pathways are tracked in the sim** — DNp02/DNp11 fire and drive DNg108, DNg97, DNge119, pIP1, AN19B001 etc., all of which have zero monitoring in the current implementation.

---

## 6. DNa02 Input Source Trace (Extended Analysis)

**Method:** `python3 trace_dn.py DNa02 --top 15`, then traced DNa02's #1 upstream partner (AN03A008).

**DNa02 upstream top 15:** AN03A008 L+R (~200 weight each), LAL018 (68–88), GNG521 (86), CB0431 (64–84), **DNa03 L+R** (70–82), **DNae005 L+R** (70–81), PS013 L+R (73–76), VES072 L+R (67–72)

**T4a, LC4, LPLC2: absent from top 15.** The visual pathway neurons that the sim injects are NOT dominant inputs to DNa02.

**DNa02 downstream top:** IN08A006 (×4), DNge026 L+R, IN19A003 L+R, **Sternal anterior rotator MN L+R** (weight 29–37), PS137, PS100, IN07B006

→ DNa02 directly innervates the sternal anterior rotator motor neurons — the wing-base muscles for flight steering. It IS the output stage.

**AN03A008 trace:**

AN03A008's downstream: **DNa02 L** (203.8), **DNa02 R** (197.2), **pIP1 L** (171.3), **pIP1 R** (145.2), then PVLP149, AVLP712m, PVLP048 (all ~30–40). Almost all output goes to DNa02 and pIP1.

AN03A008's upstream: weak distributed inputs, max weight 13.5. Notable: **LgLG6** (Lobula Giant 6, a wide-field motion detector) — weight 7.2. This IS a visual neuron!

**The actual visual pathway to DNa02:**
`T4a → (medulla local neurons) → ... → LgLG6 → AN03A008 → DNa02`

The sim injects T4a but the signal has to propagate through multiple hops (including LgLG6) before reaching AN03A008 and DNa02. At STEPS=50 ticks (~5ms), the signal likely doesn't complete this path.

**pIP1 convergence:** Both AN03A008 (weight 171) and DNp02 (weight 92) project strongly to pIP1. This is not coincidence — pIP1 appears to be a shared high-urgency maneuver neuron (its "courtship chasing" label may be an incomplete characterization; it may encode "pursue/follow" in multiple behavioral contexts).

**Why CUDA gives DNa02=5.5 and CPU gives ~0:** Likely the CUDA kernel fires neurons differently (possibly with lower threshold or different numerical integration), or there's a faster short-circuit path in the graph that the CPU advance misses within 50 ticks.

**Fix required:** To properly drive DNa02 via visual flow, the sim should inject AN03A008 or LgLG6 neurons directly (or inject deeper into the medulla pathway). The current T4a injection is too far upstream — the signal dissipates before reaching DNa02.

---

## What This Means for the Project

### Which systems are NOT doing the work
- **T4a → DNa02 (optic flow → turn):** Not encoding L/R. Both sides saturate at FLOW_GAIN≥50. The fly's turning via DNa02 is effectively symmetric bilateral noise, not directional optic flow detection.
- **LPLC2 → DNp01 (loom → escape):** Test inconclusive (fly never approached walls). Pathway not confirmed functional in normal free-flight arena conditions.

### Which systems ARE doing the work (inferred)
Based on the turn source decomposition already in fly.html HUD (added commit `2a90f84`):
- **`loom_turn`**: bilateral looming (DNp04 pathway) — the main collision-avoidance driver
- **`contact_turn`** / **`scatter_turn`**: mechanosensory + contact-DN injection — handles close-range and wall-bounce
- **`dnp04_turn`**: smooth DNp04-based looming avoidance

The fly's navigation in the current sim appears to be dominated by the loom/contact/scatter components, NOT the T4a→DNa02 directional flow pathway that the connectome is most famous for enabling.

### The FLOW_GAIN = 150 problem
The default FLOW_GAIN of 150 is in the saturation dead zone. To get L/R differential steering from T4a→DNa02, FLOW_GAIN needs to be tuned to a range where L/R input asymmetry produces L/R output asymmetry (likely somewhere in the 10–40 range, but that's where DNa02 is also silent from the gain sweep). There may be no gain setting where T4a→DNa02 encodes direction usefully — which is the core finding.

---

## Suggested Next Steps

### Immediate (next session)
1. **Rerun LPLC2 test with forced wall approach** — run brain_runner.py with arena reduced to 400×400 or manually navigate the fly toward a wall. Confirm whether LPLC2 → DNp01 actually fires and encodes direction on approach.
2. **Inject AN03A008 instead of (or in addition to) T4a** — AN03A008 is the real gateway to DNa02. One hop from DNa02 vs. 4+ hops from T4a. Add AN03A008 to `neuron_groups.json` and inject it directly with the L/R flow signal in `flow_encoder.py`.
3. **Lower FLOW_GAIN to 20–40, run on CUDA** — check if T4a→DNa02 becomes directionally sensitive in this range even if firing rates are lower overall.

### Architectural questions raised
4. **Why does CUDA give DNa02=5.5 when CPU gives 0.02?** — This is a major engine discrepancy. Either the CUDA kernel propagates differently, or the step count (50 ticks) doesn't give the CPU enough time to propagate signal to DNa02. Investigate: run CPU at steps=200 and compare.
5. **Add DNg108/DNg97/DNge119 to monitored groups** — these are downstream of DNp02/DNp11 and completely invisible to the current sim. They may be the actual motor output pathway. Add them to `neuron_groups.json` and track rates.
6. **Investigate pIP1** — appears as DNp02's #2 downstream target. Either coincidence or DNp02 has dual roles. Worth a literature check.
6. **What makes the sim "not a scripted bot"** — the HUD now shows turn decomposition. Document a run where `dn_turn` clearly drives a turn decision that `contact_turn` and `scatter_turn` would not have. This is the demo gap to close.

---

*Written by SeafoamBot research worker, 2026-10-02 ~05:30am EDT.*
*Branch: cuda-kernel. Repo: tomkirsch/fly-brain.*
