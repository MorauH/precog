"""
Usage examples for the multi-rate hierarchical PC world model.
"""

import torch
from precog.model import ClockConfig, HierarchicalClock, HierarchicalPCWorldModel


# -----------------------------------------------------------------------
# 1. Real-time robot control at 100 / 10 / 1 Hz
# -----------------------------------------------------------------------


def example_realtime_control(model: HierarchicalPCWorldModel, env):
    """Standard online control loop with real-time pacing."""

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],  # Hz per level (index 0 = fastest)
        time_scale=1.0,  # real-time
        batch_size=1,
        device="cpu",
    )

    obs, prev_action = env.reset()

    for _ in range(10_000):
        runner.clock.tick()

        result = runner.forward(obs, prev_action)

        # result.updated_levels tells you which levels fired this tick
        # e.g. tick 1   → [0]          (100 Hz level only)
        #      tick 10  → [0, 1]       (100 Hz + 10 Hz)
        #      tick 100 → [0, 1, 2]    (all three)

        obs, prev_action = env.step(result.action)

        # Block until the next 100 Hz tick wall-clock deadline
        runner.clock.sleep_until_next_tick()


# -----------------------------------------------------------------------
# 2. Fast simulation (10× faster than real-time)
# -----------------------------------------------------------------------


def example_fast_simulation(model: HierarchicalPCWorldModel, sim_env):
    """Run at 10× speed — sleep durations are 10× shorter."""

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=10.0,  # ← the only change
        batch_size=1,
        device="cuda",
    )

    obs, prev_action = sim_env.reset()

    for _ in range(100_000):
        result = runner.forward(obs, prev_action)
        obs, prev_action = sim_env.step(result.action)
        runner.clock.sleep_until_next_tick()  # sleeps 10× less


# -----------------------------------------------------------------------
# 3. Maximum-speed rollout (no pacing — as fast as hardware allows)
# -----------------------------------------------------------------------


def example_max_speed_rollout(model: HierarchicalPCWorldModel, sim_env):
    """Collect a rollout as fast as possible — skip sleep entirely."""

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=1.0,  # doesn't matter when not sleeping
        batch_size=1,
        device="cuda",
    )

    trajectory = []
    obs, prev_action = sim_env.reset()

    for _ in range(10_000):
        result = runner.forward(obs, prev_action)
        trajectory.append(
            {
                "action": result.action.cpu(),
                "surprise": result.total_surprise.item()
                if result.total_surprise
                else 0,
                "sim_time": result.sim_time,
            }
        )
        obs, prev_action = sim_env.step(result.action)
        # No sleep — run as fast as possible

    return trajectory


# -----------------------------------------------------------------------
# 4. Change simulation speed mid-episode
# -----------------------------------------------------------------------


def example_dynamic_time_scale(model: HierarchicalPCWorldModel, sim_env):
    """Slow down near interesting events, speed up during boring stretches."""

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=10.0,  # start fast
    )

    obs, prev_action = sim_env.reset()

    for step in range(50_000):
        result = runner.forward(obs, prev_action)

        # Slow to real-time when surprise is high (something unexpected)
        if result.total_surprise and result.total_surprise > 0.5:
            runner.clock.set_time_scale(1.0)
        else:
            runner.clock.set_time_scale(20.0)

        obs, prev_action = sim_env.step(result.action)
        runner.clock.sleep_until_next_tick()


# -----------------------------------------------------------------------
# 5. Introspect which levels fired
# -----------------------------------------------------------------------


def example_inspect_updates(model: HierarchicalPCWorldModel, env):
    """Show per-level update patterns and prediction errors."""

    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=1.0,
    )

    obs, prev_action = env.reset()

    for tick_i in range(200):
        result = runner.forward(obs, prev_action)
        obs, prev_action = env.step(result.action)

        # Print a summary every 100 ticks
        if tick_i % 100 == 0:
            print(
                f"tick={tick_i:4d}  "
                f"sim={result.sim_time:.3f}s  "
                f"levels_fired={result.updated_levels}  "
                f"surprise={result.total_surprise}"
            )
            for i, state in enumerate(result.level_states):
                print(
                    f"  level {i}: last_tick={state.last_update_tick}  "
                    f"err={state.last_pred_error}"
                )

        runner.clock.sleep_until_next_tick()


# -----------------------------------------------------------------------
# 6. Training (full-sequence, no rate scheduling)
# -----------------------------------------------------------------------


def example_training(model: HierarchicalPCWorldModel, dataloader, optimizer):
    """Training uses the regular forward() — no runner needed."""

    model.train()
    hidden = None

    for batch in dataloader:
        obs_dict = batch["obs"]  # (B, T, d_sensor)
        actions_gt = batch["actions"]  # (B, T, d_action)

        out = model.forward(
            obs_dict,
            prev_action=batch.get("prev_action"),
            hidden_states=hidden,
            return_all=True,
        )

        # Chain hidden states across chunks (TBPTT)
        hidden = [h.detach() for h in out["hidden_states"]]

        # Losses
        action_loss = torch.nn.functional.mse_loss(out["action_pred"], actions_gt)
        surprise_loss = out["total_surprise"]
        sigreg_loss = sum(out["sigreg_losses"])
        loss = action_loss + 0.01 * surprise_loss + 0.01 * sigreg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# -----------------------------------------------------------------------
# 7. Episode reset
# -----------------------------------------------------------------------


def example_episode_reset(model: HierarchicalPCWorldModel, env):
    runner = model.build_runner(
        level_frequencies=[100, 10, 1],
        time_scale=1.0,
    )

    for episode in range(10):
        runner.reset()  # ← zeros hidden states + clock
        obs, prev_action = env.reset()

        for _ in range(1000):
            result = runner.forward(obs, prev_action)
            obs, prev_action = env.step(result.action)
            runner.clock.sleep_until_next_tick()

        print(f"Episode {episode} done. Sim time: {runner.clock.sim_time():.2f}s")


# -----------------------------------------------------------------------
# 8. Clock standalone (use without a runner)
# -----------------------------------------------------------------------


def example_clock_standalone():
    """The clock can be used independently for any multi-rate scheduling."""

    cfg = ClockConfig(level_frequencies=[100, 10, 1], time_scale=1.0)
    clock = HierarchicalClock(cfg)

    for _ in range(110):
        clock.tick()
        flags = [clock.should_update(i) for i in range(3)]
        if any(flags):
            print(
                f"tick={clock._tick_count:3d}  "
                f"L0={'✓' if flags[0] else '·'}  "
                f"L1={'✓' if flags[1] else '·'}  "
                f"L2={'✓' if flags[2] else '·'}"
            )
