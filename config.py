"""Global configuration for NinaPro DB5 staged pre-training: MAE -> TFC -> Fine-tune."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "ninapro_db5"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset ────────────────────────────────────────────
N_SUBJECTS = 10
N_CLASSES = 53
N_CHANNELS = 16
SAMPLING_RATE = 200  # Hz
EXERCISES = ["E1", "E2", "E3"]
TRAIN_REPS = [1, 3, 4, 6]
TEST_REPS = [2, 5]

# Windowing
WINDOW_MS = 200  # ms
STRIDE_MS = 100  # ms (50% overlap)
WINDOW_LEN = int(SAMPLING_RATE * WINDOW_MS / 1000)  # 40 samples
WINDOW_STRIDE = int(SAMPLING_RATE * STRIDE_MS / 1000)  # 20 samples

# ── Encoder Architecture ───────────────────────────────
ENC_CONV_CHANNELS = [16, 128, 256, 256]  # in, c1, c2, c3
ENC_CONV_KERNELS = [5, 5, 3]
ENC_CONV_STRIDES = [1, 2, 2]
ENC_CONV_PADDING = [2, 2, 1]
ENC_DIM = 256
ENC_TRANSFORMER_LAYERS = 6
ENC_TRANSFORMER_HEADS = 8
ENC_TRANSFORMER_FFN = 1024
ENC_DROPOUT = 0.1

# ── Stage 1: MAE ───────────────────────────────────────
MAE_MASK_RATIO = 0.5
MAE_MASK_BLOCK_SIZES = [2, 4, 6, 8]  # randomly chosen per sample
MAE_EPOCHS = 300
MAE_BATCH_SIZE = 256
MAE_LR = 3e-4
MAE_WEIGHT_DECAY = 0.05
MAE_WARMUP_EPOCHS = 10

# ── Stage 2: TFC ───────────────────────────────────────
TFC_EPOCHS = 200
TFC_BATCH_SIZE = 256
TFC_LR = 1e-4
TFC_WEIGHT_DECAY = 0.05
TFC_ALPHA = 0.5  # balance contrastive vs. consistency loss
TFC_TEMPERATURE = 0.1
TFC_MARGIN = 1.0
TFC_WARMUP_EPOCHS = 5

# ── Stage 3: Fine-tune ─────────────────────────────────
FT_EPOCHS = 150
FT_BATCH_SIZE = 128
FT_LR = 1e-3
FT_WEIGHT_DECAY = 0.05
FT_WARMUP_EPOCHS = 5
FT_LABEL_SMOOTHING = 0.1
FT_DROPOUT = 0.3

# ── Wandb Logging ──────────────────────────────────────
WANDB_PROJECT = "ninapro-staged"
WANDB_ENTITY = None  # set to your wandb username/team
WANDB_MODE = "online"  # "online" | "offline" | "disabled"
WANDB_LOG_MODEL = False

# ── General ────────────────────────────────────────────
SEED = 42
NUM_WORKERS = 8
DEVICE = "cuda"
