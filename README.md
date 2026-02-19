# Crab

**An open-source bimanual robot platform for haptic-aware manipulation research.**

Crab is a low-cost bimanual robot built for studying how haptic (touch) feedback can improve learned manipulation policies. The core research question: can we train VLA policies *with* haptic sensing, then deploy them *without* — transferring haptic-informed behaviors into vision-only execution?

Built on [LeRobot](https://github.com/huggingface/lerobot) and [SmolVLA](https://huggingface.co/lerobot/smolvla_450m).

---

## Research Focus

### 1. Haptic-Aware Policy Learning

During teleoperation and training, the robot collects haptic data alongside vision:
- Force/pressure sensing in grippers
- Contact detection and slip signals
- Grasp confirmation feedback

The policy learns implicit haptic priors — grip strength modulation, contact-aware placement, slip recovery — that transfer to deployment even when haptic sensors are degraded or absent.

### 2. RL Fine-Tuning on Haptic Rewards

SmolVLA provides the base visuomotor policy. We fine-tune with reinforcement learning using haptic-derived reward signals:
- Grasp stability rewards from force feedback
- Placement precision from contact sensing
- Manipulation success from tactile confirmation

This closes the loop between touch and learned behavior without requiring haptic input at inference time.

### 3. Agentic Task Execution

A VLM-based agent layer handles task decomposition and monitoring:
- Scene understanding and step planning
- Action validation before execution
- Success verification and recovery

This gives the system multi-step task capability while the VLA handles low-level control.

---

## Architecture

```
                        "Stack the cups"
                              |
                              v
                    +---------+----------+
                    |   Agent (VLM)      |
                    |   Task planning    |
                    |   Verification     |
                    +---------+----------+
                              |
                              v
                    +---------+----------+
                    |   SmolVLA Policy   |
                    |   Haptic-trained   |
                    |   30 Hz control    |
                    +---------+----------+
                              |
                    +---------+----------+
                    |   Crab Hardware    |
                    |   2x arms + haptic |
                    |   3x cameras       |
                    +--------------------+
```

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

## Project Structure

```
examples/crab/          # Teleoperation, recording, inference scripts
src/lerobot/robots/crab/        # Robot implementation & config
src/lerobot/robots/mobile_base/ # Mobile base support
src/lerobot/motors/feetech_smart_data/  # Motor control & haptic data
src/lerobot/teleoperators/gamepad/      # Gamepad teleoperation
```

---

## Project Status

- [x] Hardware platform (Crab bimanual robot)
- [x] SmolVLA integration & edge deployment
- [x] Teleoperation with haptic data collection
- [ ] Haptic-conditioned policy training
- [ ] RL fine-tuning with haptic rewards
- [ ] Haptic-to-vision transfer evaluation
- [ ] Agentic task execution layer

---

## Why "Crab"?

Two arms. Grippers. Moves sideways sometimes.

---

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) — VLA training and robot control framework
- [SmolVLA](https://huggingface.co/lerobot/smolvla_450m) — Base visuomotor policy
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) — Vision-language model for agent layer

---

## Citation

```bibtex
@software{crab,
  title = {Crab: A Bimanual Robot Platform for Haptic-Aware Manipulation Research},
  author = {Advanced Robotic Manipulation},
  year = {2025},
  url = {https://github.com/Advanced-Robotic-Manipulation/crab}
}
```

---

## License

Apache 2.0 (following LeRobot)
