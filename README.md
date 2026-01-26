# 🦀 Crab Agent

**Hierarchical VLM-VLA control for affordable bimanual manipulation on edge hardware.**

Crab is an open-source agent architecture that brings "thinking fast and slow" to low-cost robots. A Vision-Language Model plans and reasons (System 2), while a Vision-Language-Action model executes with haptic feedback (System 1) — all running locally on a Jetson.

Built on [LeRobot](https://github.com/huggingface/lerobot).

---

## The Idea

Most robot learning research requires expensive hardware or cloud inference. We wanted to see how far we can push a **~$800 bimanual robot** with **fully on-device AI**.

The answer: structure compensates for scale.

```
┌─────────────────────────────────────────────────────────────┐
│                        CRAB AGENT                           │
│                                                             │
│   "Pick up the red cup"                                     │
│            │                                                │
│            ▼                                                │
│   ┌─────────────────┐         ┌─────────────────┐          │
│   │   SYSTEM 2      │         │   SYSTEM 1      │          │
│   │   (VLM)         │────────▶│   (VLA)         │          │
│   │                 │         │                 │          │
│   │ • Observe scene │         │ • 30Hz control  │          │
│   │ • Plan actions  │         │ • Haptic feedback│         │
│   │ • Verify result │         │ • Action chunks │          │
│   │                 │         │                 │          │
│   │ Qwen3-VL-8B     │         │ SmolVLA 450M    │          │
│   │ ~2-3 sec/step   │         │ ~1-2 sec/action │          │
│   └─────────────────┘         └─────────────────┘          │
│                                                             │
│            Jetson Orin NX 16GB — No cloud needed            │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Ideas

### 1. Validated Plan-ReAct

The VLM doesn't just output actions — it plans multiple steps, but only commits to one. Before execution, rule-based validation catches errors. No bad moves reach the robot.

```
OBSERVE → PLAN → VALIDATE → EXECUTE → VERIFY
             │        │
             │    rejection
             │        │
             └────────┘
              (retry with feedback)
```

### 2. Schema-Guided Reasoning (SGR)

Small models need structure. We use constrained decoding to force the VLM through a reasoning checklist:

```
Plan:
  current_state: "Holding red cup, plate visible"
  remaining_steps: ["place cup", "pick next", "stack"]
  reasoning: "Cup in hand, plate clear"
  confidence: 8
  action: Place { location: "blue plate", hand: "left" }
```

The schema *is* the prompt engineering.

### 3. Haptic-in-the-Loop

SmolVLA doesn't just replay trajectories — it responds to touch:
- Grasp confirmation via force threshold
- Slip detection triggers grip adjustment  
- Contact sensing for placement
- Collision detection stops motion

---

## Hardware

| Component | Spec | Cost |
|-----------|------|------|
| Compute | Jetson Orin NX 16GB | ~$500 |
| Arms | 2x SO-101 (6 DOF each) | ~$200 |
| Grippers | With haptic sensors | ~$50 |
| Cameras | 3x RGB (wrist + overhead) | ~$50 |
| **Total** | | **~$800** |

---

## Models

| Role | Model | Memory | Speed |
|------|-------|--------|-------|
| Planner | Qwen3-VL-8B INT4 | ~6 GB | ~15 tok/s |
| Executor | SmolVLA 450M FP16 | ~1.5 GB | 30 Hz |

Both run on device. Serialized execution (not simultaneous) keeps memory in check.

---

## Project Status

🚧 **Work in progress** 

- [x] Architecture design
- [x] Model selection & memory planning
- [ ] SGR schema implementation
- [ ] Validation rules
- [ ] SmolVLA + haptic integration
- [ ] Real hardware testing
- [ ] Training data collection
- [ ] Fine-tuning for our setup

---

## Architecture

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full specification.

**The loop:**

1. **OBSERVE** — VLM describes scene as structured JSON
2. **PLAN** — VLM outputs action using SGR schema (Plan-ReAct pattern)
3. **VALIDATE** — Rule-based checks before execution (no LLM)
4. **EXECUTE** — SmolVLA runs action with haptic feedback
5. **VERIFY** — VLM confirms success or triggers recovery

**Timing:** ~5-10 seconds per subtask. Not fast, but reliable.

---

## Why "Crab"?

Two arms. Grippers. Moves sideways sometimes. 🦀

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — Foundation for VLA training and deployment
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — VLM with 3D spatial reasoning
- [SmolVLA](https://huggingface.co/lerobot/smolvla_450m) — Compact VLA that fits on edge
- [SGR](https://abdullin.com/schema-guided-reasoning/) — Schema-Guided Reasoning patterns
- [ERC-3 Winners](https://github.com/IlyaRice/Enterprise-RAG-Challenge-3-AI-Agents) — Validation and agent loop patterns

---

## Citation

```bibtex
@software{crab-agent,
  title = {Crab Agent: Hierarchical VLM-VLA Control for Affordable Bimanual Manipulation},
  author = {TODO},
  year = {2025},
  url = {https://github.com/TODO/crab-agent}
}
```

---

## License

MIT
