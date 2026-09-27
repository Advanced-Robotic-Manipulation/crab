# HapticVLA: IROS 2026 Lightning Talk, speaker script

**Slot:** Lightning session, IROS 2026, Pittsburgh (Sept 27 – Oct 1).
**Time:** 2:50 in total, and the chair's introduction counts toward it. You get a warning at **2:20** and a hard cut at **3:00**. There is no Q&A; the talk is an advert for the poster.
**Deck:** `HapticVLA_IROS2026_Lightning.pptx` (7 slides, 16:9). The same script is in the speaker notes.
**Length:** about 320 words. At a relaxed pace, with pauses, that is roughly 2:20–2:30.
**Source:** checked line by line against the paper's LaTeX source (`Advanced-Robotic-Manipulation/posters`, `hapticvla/paper/`). Every number and claim on the slides comes from it.

---

## Memory map: spell **T · O · U · C · H**

Five letters in the top-right corner of each content slide show where you are. You don't memorise sentences. You remember one word per letter and then talk about what's on the slide.

| Letter | Say this slide as… | What's on screen to talk from | Bridge to next |
|---|---|---|---|
| (title) | "Hand a robot an egg…" | Robot + title | "Here's the trouble…" |
| **T** | **Trouble**: robots see, they don't feel | Two cards = two bad options | Read the bottom line |
| **O** | **One idea**: touch in training, not deployment | Lab panel, then anywhere panel | "Two steps. Step one…" |
| **U** | **Use touch as a reward** | Architecture figure + 3 chips: Score, Penalize, Reward | "The purple box is step two…" |
| **C** | **Cut the sensors**: it gets better | Teacher, then student + bar chart | "And yes, it holds the egg." |
| **H** | **Hold the egg** | Looping egg video + 3 result rows | "So…" |
| (close) | Train with touch. Deploy without it. | QR codes | "Thank you!" |

**The story thread:** you open with an egg and you close with the egg.

**Number hooks:**
- **57 → 75 → 87** (mean success: base, then teacher, then HapticVLA). 57 and 75 are the same digits flipped, then add 12 more.
- **Jar 75, Waffles 90, Egg 95.** Top to bottom, the numbers only go up, and the most fragile object has the biggest jump: egg **+45**.
- **86.7 / +30 / 0** is the bottom row on slide H: mean success, gain over the base model, and sensors at deployment.
- **Goldilocks grip** for slide U: the reward punishes squeezing too hard, too loose, or slipping.

---

## Script (with timestamps)

> Timestamps count from the moment **you** start talking (the chair intro takes about 10 s before that).

**1 · Title** (0:00 – 0:10)
"Hand a robot an egg. Will it crack it? Our robot learned to be gentle, and at deployment it doesn't even need to feel the egg."

**2 · T: Trouble** (0:10 – 0:30)
"Here's the trouble. Vision-language-action models only *see*. A camera shows where the egg is, not how hard you're squeezing it. So they crush fragile things. The classic fix is tactile sensors, but they're expensive, and they tie your policy to special hardware.
*(read the bottom line)* So what if a robot could learn the feel… and leave the sensors behind?"

**3 · O: One idea** (0:30 – 0:50)
"That's our one idea: touch at *training* time, not at deployment. We learn on a robot that can feel, then ship a policy that needs only cameras. Touch becomes a teacher, not a part you have to install."

**4 · U: Use touch as a reward** (0:50 – 1:25)
"Step one: use touch as a reward. We start from SmolVLA, a small vision-language model plus an action expert that outputs motions. Offline, the fingertip sensors score every moment of every demo. Too hard, too loose, or slipping: penalty. Gentle and successful: bonus. We even recorded failures on purpose, so it learns what 'too hard' looks like. The action expert then learns more from the gentle moves, and less from the crushing ones."
*(Optional name-drop: "We call it Safety-Aware Reward-Weighted Flow Matching.")*
*(From the paper: the 310 demos are 240 successful ones plus 70 deliberate failures, where too much force damaged the object. The failures get a negative reward.)*

**5 · C: Cut the sensors** (1:25 – 1:55)
"Step two: cut the sensors. The touch-trained teacher labels the data once, offline. The student is the same small model with no tactile input, and it learns from a 50/50 mix of demos and teacher actions. And here's the twist: the student that can't feel **beats** the teacher that can, 86.7 versus 75 percent." *(pause)*
*(If you need a why: "It learns to act as if it could feel. The teacher's actions already carry the touch.")*

**6 · H: Hold the egg** (1:55 – 2:25). The 2:20 warning lands here.
"And yes, it holds the egg. Real robot, three delicate tasks, 20 trials each. Jar: 50 to 75 percent. Waffles: 70 to 90. And the egg: 50 to 95. That's plus 30 points on average, with zero tactile sensors and no extra compute at deployment."
*If you're running late, say only: "Egg: 50 to 95. Plus 30 points on average, zero sensors."*

**7 · Close** (2:25 – 2:35)
"Train with touch, deploy without it. Code, data and models are open: scan the QR codes, and come see the robot at our poster. Thank you!"
*(If you know your poster board number, say it here.)*

---

## If someone catches you at the poster (all from the paper)

- **"What about bigger VLAs?"** X-VLA (0.9B, large cross-embodiment pretraining) and VLA-0 (bigger, and it previously beat SmolVLA on the SO-101) were fine-tuned on the same data. Both scored 0% on all three tasks. The paper reads this as a failure to generalize to high-precision, contact-rich tasks.
- **"Does the tactile teacher ever win?"** The best tactile-at-inference variant (async teacher) reached 81.7%. That is still below the sensor-free student's 86.7%. Async hurts because tactile readings and actions fall out of sync (latency).
- **"What exactly is penalized?"**
  - The per-step penalties cover: over-force, under-force while holding, peak pressure, pressure concentration, left/right pad asymmetry, and slip. Slip means a centre-of-pressure jump or a sudden force drop.
  - Episode reward = step reward + success − drop − damage − risk.
  - All thresholds are calibrated from dataset quantiles.
- **"How is it weighted?"**
  - The advantage mixes the episode score and the chunk score as robust z-scores, 0.7 and 0.3.
  - Weight = exp(0.25·A), clipped to [0.25, 4] and renormalized per batch.
  - An L2 anchor to the imitation model prevents drift or mode collapse.
- **"How does distillation work?"**
  - Teacher action chunks are pre-computed once, fully offline.
  - The student is initialised from the teacher without the tactile encoder; the state projection is sliced from 134 inputs to 6.
  - It trains on 0.5·demo + 0.5·teacher targets, with zero architecture changes.
- **Data:** 310 real episodes. Jar and Waffles have 70 good + 20 failed each; the bimanual Egg task has 100 + 30.
- **Hardware:**
  - Two LeRobot SO-101 arms. The right arm has 12 V servos and a parallel gripper with a 10×10 taxel array per finger: 200 taxels, 1–9 N, 120 Hz. The sensors came from Prof. Kajimoto's lab at UEC.
  - Cameras: a RealSense D435 plus IMX335 wrist cameras, at 640×480.
  - Everything runs on a Jetson Orin NX 16 GB.
- **"Isn't this FD-VLA?"** FD-VLA distils *force* (from force sensors) into a token. Gano et al. pre-train with low-fidelity tactile and then drop it, but that is not a VLA. HapticVLA does it for *touch*, inside a VLA, with a tactile safety reward. The paper's Fig. 1 caption calls it "the first VLA model to enable tactile-aware manipulation without requiring force sensors at inference". If you use "first", phrase it that way.

## Before you walk up: checklist

1. **Embed the fonts.** The deck uses the poster fonts: Instrument Serif, IBM Plex Sans and JetBrains Mono. IROS asks you to embed fonts because you'll present from the ready-room PC.
   - Open the deck on a machine that has those fonts installed.
   - Windows: *File → Options → Save → "Embed fonts in the file" → "Embed all characters"*, then save.
   - Mac: *PowerPoint → Preferences → Save → "Embed fonts in the file"*, then save.
2. **Upload** the `.pptx` at the Speaker Ready Room before your session, and check that it opens on their computer. The egg clip on slide H is an animated GIF inside the file, so it plays without any links.
3. **Backup:** `HapticVLA_IROS2026_Lightning.pdf` has every font embedded. If PowerPoint misbehaves, present it full-screen from any PDF viewer. The GIF shows as a still frame there.
4. **Walk up** during the last 10 s of the previous talk. You don't need a clicker warm-up: the first slide is already up while the chair introduces you.
