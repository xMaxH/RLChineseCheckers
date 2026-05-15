"""Hyperparameters and per-stage configs for AlphaZero training."""

from dataclasses import dataclass, field
from typing import Tuple

# ---------- Game / board constants (must match game.py) ----------
COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']
COLOUR_OPPOSITES = {
    'red': 'blue', 'blue': 'red',
    'lawn green': 'gray0', 'gray0': 'lawn green',
    'yellow': 'purple', 'purple': 'yellow',
}
PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}
NUM_CELLS = 121
PINS_PER_PLAYER = 10
BOARD_CHANNELS = 6 + 6 + 1 + 1 + PINS_PER_PLAYER
NUM_ACTIONS = PINS_PER_PLAYER * NUM_CELLS  # 1210
MAX_PLAYERS = 6


# ---------- Network ----------
@dataclass(frozen=True)
class NetConfig:
    in_channels: int = BOARD_CHANNELS
    global_dim: int = 8
    width: int = 192
    blocks: int = 8
    policy_head_channels: int = 16
    value_hidden: int = 128
    num_actions: int = NUM_ACTIONS
    num_player_slots: int = MAX_PLAYERS  # 6-vector value head


# ---------- MCTS ----------
@dataclass(frozen=True)
class MCTSConfig:
    n_sim: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    virtual_loss: float = 1.0
    batch_leaves: int = 32
    add_noise: bool = True   # only at root during self-play


# ---------- Self-play ----------
@dataclass(frozen=True)
class SelfPlayConfig:
    num_workers: int = 4
    games_per_chunk: int = 200
    max_moves_2p: int = 200
    max_moves_multi: int = 300  # for 4p+
    moves_per_player: int = 0   # if >0, overrides above: max_moves = moves_per_player * num_players
    candidate_slot_frac: float = 0.50
    heuristic_slot_frac: float = 0.25
    snapshot_slot_frac: float = 0.25
    snapshot_pool_size: int = 10
    snapshot_every_train_steps: int = 5
    dagger_in_bootstrap: bool = False
    dagger_policy_temperature: float = 0.0
    heuristic_rollout_targets: bool = False
    heuristic_rollouts_per_move: int = 1
    heuristic_rollout_pool_cap: int = 12
    heuristic_rollout_score_temperature: float = 250.0
    inference_batch_max: int = 128
    inference_batch_timeout_ms: float = 5.0


# ---------- Training ----------
@dataclass(frozen=True)
class TrainConfig:
    replay_capacity: int = 200_000
    min_samples_to_train: int = 50_000
    sample_per_step: int = 8192
    batch_size: int = 4096
    lr: float = 1e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    entropy_bonus: float = 0.01
    epochs_per_chunk: int = 1
    eval_every_steps: int = 50
    eval_games: int = 100
    log_every_steps: int = 1
    cosine_total_steps: int = 0  # 0 = no cosine schedule
    min_train_steps: int = 4     # always run at least this many gradient steps per chunk


# ---------- Curriculum stage spec ----------
@dataclass(frozen=True)
class StageSpec:
    name: str
    player_counts: Tuple[int, ...]
    player_count_weights: Tuple[float, ...]
    pass_winrate: float           # vs heuristic
    max_wallclock_hours: float
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------- Concrete stage configs ----------
def verify_stage() -> StageSpec:
    return StageSpec(
        name="verify_2p",
        player_counts=(2,),
        player_count_weights=(1.0,),
        pass_winrate=0.80,
        max_wallclock_hours=2.0,
        selfplay=SelfPlayConfig(games_per_chunk=40, num_workers=4),
        train=TrainConfig(
            replay_capacity=200_000,
            min_samples_to_train=10_000,  # smaller for fast verification
            sample_per_step=8192,
            batch_size=4096,
            lr=1e-4,
            eval_every_steps=20,
            eval_games=40,
        ),
    )


def overnight_stage_2p() -> StageSpec:
    return StageSpec(
        name="overnight_2p",
        player_counts=(2,),
        player_count_weights=(1.0,),
        pass_winrate=0.80,
        max_wallclock_hours=12.0,
        selfplay=SelfPlayConfig(games_per_chunk=200),
        train=TrainConfig(replay_capacity=1_000_000,
                          min_samples_to_train=200_000,
                          sample_per_step=8192,
                          batch_size=4096),
    )


def stage_2_3p() -> StageSpec:
    return StageSpec(
        name="curriculum_2_3p",
        player_counts=(2, 3),
        player_count_weights=(0.5, 0.5),
        pass_winrate=0.65,
        max_wallclock_hours=12.0,
        selfplay=SelfPlayConfig(games_per_chunk=200),
        train=TrainConfig(replay_capacity=1_000_000,
                          min_samples_to_train=200_000),
    )


def stage_2_4p() -> StageSpec:
    return StageSpec(
        name="curriculum_2_4p",
        player_counts=(2, 3, 4),
        player_count_weights=(0.4, 0.3, 0.3),
        pass_winrate=0.50,
        max_wallclock_hours=12.0,
        selfplay=SelfPlayConfig(games_per_chunk=200),
        train=TrainConfig(replay_capacity=1_500_000,
                          min_samples_to_train=300_000),
    )


def stage_2_5p() -> StageSpec:
    return StageSpec(
        name="curriculum_2_5p",
        player_counts=(2, 3, 4, 5),
        player_count_weights=(0.3, 0.25, 0.25, 0.2),
        pass_winrate=0.40,
        max_wallclock_hours=12.0,
        selfplay=SelfPlayConfig(games_per_chunk=200),
        train=TrainConfig(replay_capacity=2_000_000,
                          min_samples_to_train=400_000),
    )


def stage_2_6p() -> StageSpec:
    return StageSpec(
        name="curriculum_2_6p",
        player_counts=(2, 3, 4, 5, 6),
        player_count_weights=(0.25, 0.2, 0.2, 0.2, 0.15),
        pass_winrate=0.35,
        max_wallclock_hours=24.0,
        selfplay=SelfPlayConfig(games_per_chunk=200),
        train=TrainConfig(replay_capacity=4_000_000,
                          min_samples_to_train=800_000,
                          sample_per_step=65536,
                          batch_size=32768,
                          lr=3e-5),
    )


CURRICULUM = [
    verify_stage(),
    overnight_stage_2p(),
    stage_2_3p(),
    stage_2_4p(),
    stage_2_5p(),
    stage_2_6p(),
]
