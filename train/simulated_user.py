"""
SimulatedUserAgent
===================
Simulates a human learner for the Emotiv Learn adaptive tutor loop.

Per content step t, given content_text, produces:
  - eeg_features_t   : normalized bandpower feature vector (float array, shape [62])
  - behavioral_cues_t: dict {time_on_chunk, scroll_rate, reread_count}
  - next_action_t    : one of {"continue", "clarify", "branch"}
  - next_prompt_t    : user prompt string (LLM-generated for clarify/branch, None for continue)
  - cognitive_load   : float in [0, 1]  (debug)
  - epochs_consumed  : int              (debug)

STEW dataset assumptions
------------------------
- .txt files under stew_dir/, named sub01_hi.txt, sub02_hi.txt, ...
- Each file is tab- or comma-separated: one row per sample, 14 columns (EEG channels)
- lo files are ignored entirely
- A separate ratings file (ratings.txt by default) maps subject → workload score 1–9
  Format: one subject per line, e.g.  sub01    6.0
- Only the hi condition is used (rest condition excluded by design)
- 30-second epochs extracted with no overlap → 30 × 128 = 3840 samples per epoch
"""

import os
import json
import asyncio
import numpy as np
from scipy.signal import welch
from dotenv import load_dotenv

load_dotenv()


# ── Constants ─────────────────────────────────────────────────────────────────

FS            = 128
EPOCH_SEC     = 30
EPOCH_SAMPLES = FS * EPOCH_SEC   # 3840

CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
IDX_F3   = CHANNELS.index("F3")
IDX_F4   = CHANNELS.index("F4")

BANDS = {
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

ACTIONS       = ["continue", "clarify", "branch"]
BASE_TIME_SEC = 45.0
MAX_TIME_SEC  = 180.0

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
PROMPT_MODEL   = "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ── EEG helpers ───────────────────────────────────────────────────────────────

def _bandpower(signal_1d: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nperseg = min(len(signal_1d), fs * 4)
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(psd[mask]))


def _extract_features(epochs: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract 62-dim feature vector from one or more epochs.

    Layout:
      [0:56]  relative band power  (4 bands x 14 channels, flattened row-major)
      [56:58] frontal alpha asymmetry  (relative: alpha_F4 - alpha_F3, absolute version)
      [58:62] frontal theta/alpha ratio    (F7, F3, F4, F8)
    """
    data = epochs.mean(axis=0)   # (EPOCH_SAMPLES, 14)

    abs_bp = np.zeros((len(BANDS), 14))
    for b_idx, (_, (flo, fhi)) in enumerate(BANDS.items()):
        for ch in range(14):
            abs_bp[b_idx, ch] = _bandpower(data[:, ch], fs, flo, fhi)

    total_power = abs_bp.sum(axis=0, keepdims=True) + 1e-12
    rel_bp = abs_bp / total_power   # (4, 14)

    asymmetry     = rel_bp[1, IDX_F4] - rel_bp[1, IDX_F3]
    asymmetry_abs = abs_bp[1, IDX_F4] - abs_bp[1, IDX_F3]

    frontal_ch        = [CHANNELS.index(c) for c in ["F7","F3","F4","F8"]]
    theta_alpha_ratio = rel_bp[0, frontal_ch] / (rel_bp[1, frontal_ch] + 1e-12)

    return np.concatenate([
        rel_bp.flatten(),            # 56
        [asymmetry, asymmetry_abs],  # 2
        theta_alpha_ratio,           # 4
    ]).astype(np.float32)


def _cognitive_load_score(features: np.ndarray) -> float:
    """Sigmoid-squashed scalar load score in [0,1] from frontal theta/alpha ratio."""
    theta_alpha = features[58:62].mean()
    normalized  = (theta_alpha - 0.5) / 2.5
    score       = 1.0 / (1.0 + np.exp(-4.0 * (normalized - 0.5)))
    return float(np.clip(score, 0.0, 1.0))


def _estimate_content_complexity(text: str) -> float:
    """Heuristic complexity in [0,1] based on sentence length and vocab diversity."""
    sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
    words     = text.lower().split()
    if not sentences or not words:
        return 0.5
    avg_sent_len    = np.mean([len(s.split()) for s in sentences])
    vocab_diversity = len(set(words)) / len(words)
    sent_score  = float(np.clip((avg_sent_len - 8) / 22, 0, 1))
    vocab_score = float(np.clip((vocab_diversity - 0.3) / 0.6, 0, 1))
    return 0.6 * sent_score + 0.4 * vocab_score


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ── STEW Loader (.txt version) ────────────────────────────────────────────────

class STEWLoader:
    """
    Loads STEW hi-condition .txt files and a separate ratings file.

    Directory structure:
        stew_dir/
            sub01_hi.txt
            sub02_hi.txt
            ...
            ratings.txt

    ratings.txt format (comma-separated, no header):
        subject_number, rest_rating, hi_rating
        e.g.  1, 2, 8
        Subjects 5, 24, and 42 are missing — they get NaN rating automatically.
        Only the hi_rating (3rd column) is used.

    Each *_hi.txt: rows = samples, 14 numeric columns = EEG channels at 128 Hz.
    Delimiter auto-detected (tab or comma).
    lo files are ignored.
    """

    def __init__(self, stew_dir: str, ratings_file: str = "ratings.txt"):
        self.stew_dir  = stew_dir
        self._ratings  = self._load_ratings(os.path.join(stew_dir, ratings_file))
        self._subjects = self._discover_subjects()

    def _load_ratings(self, path: str) -> dict:
        """
        Parse ratings.txt. Format per line:  subject_num, rest_rating, hi_rating
        Keys stored as zero-padded subject IDs matching filenames, e.g. "sub01".
        Only the hi_rating (column index 2) is kept.
        """
        ratings = {}
        if not os.path.exists(path):
            return ratings
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    subj_num  = int(parts[0])
                    hi_rating = float(parts[2])
                    subj_id   = f"sub{subj_num:02d}"   # 1 → "sub01", 24 → "sub24"
                    ratings[subj_id] = hi_rating
                except ValueError:
                    continue   # skip malformed lines
        return ratings

    def _discover_subjects(self) -> dict:
        subjects = {}
        for fname in sorted(os.listdir(self.stew_dir)):
            if fname.endswith("_hi.txt"):
                subj_id = fname.replace("_hi.txt", "")
                subjects[subj_id] = os.path.join(self.stew_dir, fname)
        if not subjects:
            raise FileNotFoundError(f"No *_hi.txt files found in {self.stew_dir}")
        return subjects

    def _detect_delimiter(self, path: str):
        with open(path) as f:
            first_line = f.readline()
        if "\t" in first_line:
            return "\t"
        elif "," in first_line:
            return ","
        else:
            return None  # None tells np.loadtxt to split on any whitespace

    @property
    def subject_ids(self) -> list:
        return list(self._subjects.keys())

    def load_subject(self, subject_id: str) -> tuple:
        """
        Returns (epochs, rating).
            epochs : np.ndarray (n_epochs, EPOCH_SAMPLES, 14)
            rating : float in [1, 9]  (NaN if not in ratings file)
        """
        path      = self._subjects[subject_id]
        delimiter = self._detect_delimiter(path)
        raw       = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)

        if raw.ndim != 2 or raw.shape[1] != 14:
            raise ValueError(
                f"{path}: expected shape (n_samples, 14), got {raw.shape}"
            )

        n_epochs = len(raw) // EPOCH_SAMPLES
        if n_epochs == 0:
            raise ValueError(
                f"{path}: not enough samples for even one {EPOCH_SEC}s epoch "
                f"(need {EPOCH_SAMPLES}, got {len(raw)})"
            )

        trimmed = raw[: n_epochs * EPOCH_SAMPLES]
        epochs  = trimmed.reshape(n_epochs, EPOCH_SAMPLES, 14)
        rating  = self._ratings.get(subject_id, float("nan"))

        return epochs, rating


# ── LLM prompt generator ──────────────────────────────────────────────────────

async def _generate_next_prompt(
    content_text: str,
    action: str,
    cognitive_load: float,
) -> str | None:
    """
    Call the OpenAI API to generate a realistic student prompt for clarify/branch.
    Returns None for 'continue'.
    Falls back to a template string if the API call fails.
    """
    if action == "continue":
        return None

    try:
        import urllib.request

        system_message = (
            "You are simulating a student interacting with an adaptive learning app. "
            "Given a content chunk and the student's current action, generate a short, "
            "realistic student message (1-2 sentences). "
            "Respond with ONLY the student message — no preamble, no quotes."
        )

        if action == "clarify":
            instruction = (
                f"The student (cognitive load: {cognitive_load:.2f}/1.0) found this confusing "
                f"and wants clarification. Generate their clarify message."
            )
        else:  # branch
            instruction = (
                f"The student (cognitive load: {cognitive_load:.2f}/1.0) understood the content "
                f"and wants to explore a related tangent or go deeper on one part. "
                f"Generate their branch/exploration message."
            )

        payload = json.dumps({
            "model": PROMPT_MODEL,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": f"Content chunk:\n{content_text}\n\nInstruction: {instruction}",
                }
            ],
        }).encode()

        req = urllib.request.Request(
            OPENAI_API_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        message_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return message_content.strip() if message_content else None

    except Exception as e:
        fallback = {
            "clarify": "Can you explain that differently? I'm not sure I followed.",
            "branch":  "Interesting — can we explore that idea a bit further?",
        }
        print(f"[SimulatedHumanAgent] LLM prompt generation failed ({e}), using fallback.")
        return fallback.get(action)


# ── SimulatedHumanAgent ───────────────────────────────────────────────────────

class SimulatedHumanAgent:
    """
    Simulates a human learner interacting with the adaptive tutor.

    Parameters
    ----------
    stew_dir     : path to directory containing sub##_hi.txt files + ratings.txt
    subject_id   : which subject to replay (e.g. "sub01"). None = random.
    ratings_file : filename of the ratings file inside stew_dir
    seed         : random seed
    loop_epochs  : if True, loop when epochs exhausted; if False, advance to next subject
    """

    def __init__(
        self,
        stew_dir: str,
        subject_id: str | None = None,
        ratings_file: str = "ratings.txt",
        seed: int | None = None,
        loop_epochs: bool = True,
    ):
        self.rng         = np.random.default_rng(seed)
        self.loop_epochs = loop_epochs
        self.loader      = STEWLoader(stew_dir, ratings_file)
        self._load_subject(subject_id or self.rng.choice(self.loader.subject_ids))

    # ── subject management ────────────────────────────────────────────────────

    def _load_subject(self, subject_id: str):
        self.subject_id          = subject_id
        self.epochs, self.rating = self.loader.load_subject(subject_id)
        self.cursor              = 0
        self._subj_list          = self.loader.subject_ids
        self._subj_idx           = self._subj_list.index(subject_id)

    def _advance_subject(self):
        self._subj_idx = (self._subj_idx + 1) % len(self._subj_list)
        self._load_subject(self._subj_list[self._subj_idx])

    def _sample_epochs(self, n: int) -> np.ndarray:
        """Pull n consecutive epochs, looping or switching subject on exhaustion."""
        collected = []
        remaining = n
        while remaining > 0:
            available = len(self.epochs) - self.cursor
            take      = min(remaining, available)
            collected.append(self.epochs[self.cursor : self.cursor + take])
            self.cursor += take
            remaining   -= take
            if remaining > 0:
                if self.loop_epochs:
                    self.cursor = 0
                else:
                    self._advance_subject()
        return np.concatenate(collected, axis=0)

    # ── core step ─────────────────────────────────────────────────────────────

    def step(self, content_text: str) -> dict:
        """
        Synchronous step. Returns:

        {
            eeg_features    : np.ndarray (62,)
            behavioral_cues : {time_on_chunk, scroll_rate, reread_count}
            next_action     : "continue" | "clarify" | "branch"
            next_prompt     : str | None  (None when next_action == "continue")
            cognitive_load  : float
            epochs_consumed : int
        }
        """
        return asyncio.run(self.async_step(content_text))

    async def async_step(self, content_text: str) -> dict:
        """Async version of step() — use this when running inside an existing async loop."""
        content_complexity = _estimate_content_complexity(content_text)

        # Pass 1 — seed epoch for initial load estimate
        seed_epoch    = self._sample_epochs(1)
        seed_features = _extract_features(seed_epoch)
        load_estimate = _cognitive_load_score(seed_features)

        # Compute time on chunk → number of epochs to consume
        time_on_chunk = self._simulate_time(load_estimate, content_complexity)
        n_epochs      = max(1, round(time_on_chunk / EPOCH_SEC))

        # Pass 2 — consume full window (seed epoch already consumed)
        if n_epochs > 1:
            extra      = self._sample_epochs(n_epochs - 1)
            all_epochs = np.concatenate([seed_epoch, extra], axis=0)
        else:
            all_epochs = seed_epoch

        final_features = _extract_features(all_epochs)
        final_load     = _cognitive_load_score(final_features)

        behavioral_cues = self._simulate_behavioral_cues(
            final_load, content_complexity, time_on_chunk
        )
        next_action = self._sample_action(final_load, behavioral_cues)
        next_prompt = await _generate_next_prompt(content_text, next_action, final_load)

        return {
            "eeg_features":    final_features,
            "behavioral_cues": behavioral_cues,
            "next_action":     next_action,
            "next_prompt":     next_prompt,
            "cognitive_load":  final_load,
            "epochs_consumed": n_epochs,
        }

    # ── behavioral simulation ─────────────────────────────────────────────────

    def _simulate_time(self, cognitive_load: float, content_complexity: float) -> float:
        mean_time = BASE_TIME_SEC * (1 + 0.5 * cognitive_load) * (1 + 0.4 * content_complexity)
        mean_time = float(np.clip(mean_time, EPOCH_SEC, MAX_TIME_SEC))
        noise     = self.rng.normal(0, mean_time * 0.15)
        return float(np.clip(mean_time + noise, EPOCH_SEC, MAX_TIME_SEC))

    def _simulate_behavioral_cues(
        self,
        cognitive_load: float,
        content_complexity: float,
        time_on_chunk: float,
    ) -> dict:
        scroll_rate = float(np.clip(
            1.0 - 0.5 * cognitive_load - 0.3 * content_complexity
            + self.rng.normal(0, 0.05),
            0.05, 1.0,
        ))
        struggle     = 0.6 * cognitive_load + 0.4 * content_complexity
        reread_count = int(self.rng.poisson(max(0.0, struggle * 3.0)))
        return {
            "time_on_chunk": round(time_on_chunk, 1),
            "scroll_rate":   round(scroll_rate, 3),
            "reread_count":  reread_count,
        }

    def _sample_action(self, cognitive_load: float, behavioral_cues: dict) -> str:
        scroll_rate  = behavioral_cues["scroll_rate"]
        reread_count = behavioral_cues["reread_count"]
        struggle = (
            0.45 * cognitive_load
            + 0.30 * (1.0 - scroll_rate)
            + 0.25 * np.tanh(reread_count / 3.0)
        )
        curiosity_boost = 0.3 * (1 - abs(cognitive_load - 0.4))
        logits = np.array([
            2.0 - 3.0 * struggle,    # continue
           -1.0 + 4.0 * struggle,    # clarify
           -1.5 + curiosity_boost,   # branch
        ])
        probs = _softmax(logits)
        return str(self.rng.choice(ACTIONS, p=probs))

    # ── utilities ─────────────────────────────────────────────────────────────

    def reset(self, subject_id: str | None = None):
        """Reset cursor, optionally switching to a different subject."""
        self._load_subject(subject_id or self.subject_id)

    def __repr__(self):
        return (
            f"SimulatedHumanAgent(subject={self.subject_id}, "
            f"cursor={self.cursor}/{len(self.epochs)} epochs, "
            f"rating={self.rating})"
        )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    stew_dir = sys.argv[1] if len(sys.argv) > 1 else "../stew_dataset/"

    agent = SimulatedHumanAgent(stew_dir=stew_dir, seed=42)
    print(agent)

    test_contents = [
        "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared.",
        """
        Backpropagation computes gradients by applying the chain rule recursively
        through the computational graph. For each layer l, the gradient of the loss
        with respect to weights W_l is dL/dW_l = dL/da_l times da_l/dW_l,
        where a_l is the pre-activation. This requires storing intermediate activations
        during the forward pass, which is why memory scales with network depth.
        """,
        "Gradient descent minimizes a loss function by stepping opposite to the gradient.",
    ]

    print("\n--- Simulating 3 content steps ---\n")
    for i, content in enumerate(test_contents):
        result = agent.step(content)
        print(f"Step {i+1}:")
        print(f"  epochs consumed : {result['epochs_consumed']}")
        print(f"  cognitive load  : {result['cognitive_load']:.3f}")
        print(f"  time on chunk   : {result['behavioral_cues']['time_on_chunk']}s")
        print(f"  scroll rate     : {result['behavioral_cues']['scroll_rate']}")
        print(f"  reread count    : {result['behavioral_cues']['reread_count']}")
        print(f"  next action     : {result['next_action']}")
        print(f"  next prompt     : {result['next_prompt']}")
        print(f"  eeg shape       : {result['eeg_features'].shape}")
        print()
