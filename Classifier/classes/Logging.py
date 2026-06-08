from rich.table import Table
from rich.panel import Panel
from rich.box import SIMPLE
from rich.text import Text


class TrainingDashboard:

    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

    def _sparkline(self, values, width=30):
        if len(values) < 2:
            return "▁"

        blocks = "▁▂▃▄▅▆▇█"
        v = values[-width:]

        v_min, v_max = min(v), max(v)
        denom = (v_max - v_min) + 1e-12

        scaled = [
            int((x - v_min) / denom * (len(blocks) - 1))
            for x in v
        ]

        return "".join(blocks[i] for i in scaled)

    def _sparkline_delta(self, values, width=30):
        """
        Oscillator-style delta sparkline (zero-centered signal).

        - green = improving (below zero)
        - red   = worsening (above zero)
        - intensity = magnitude of change
        - EMA smoothing for stability
        """

        if len(values) < 2:
            return [("─", "dim")]

        # -----------------------
        # raw deltas
        # -----------------------
        deltas = [
            values[i] - values[i - 1]
            for i in range(1, len(values))
        ]

        # -----------------------
        # EMA smoothing (oscillator stability)
        # -----------------------
        alpha = 0.3
        smoothed = []
        ema = deltas[0]

        for d in deltas:
            ema = alpha * d + (1 - alpha) * ema
            smoothed.append(ema)

        smoothed = smoothed[-width:]

        max_abs = max(abs(x) for x in smoothed) + 1e-12

        # intensity levels (oscillator bands)
        levels = "▁▂▃▄▅▆▇█"

        rendered = []

        for v in smoothed:

            mag = abs(v) / max_abs
            idx = int(mag * (len(levels) - 1))
            idx = min(idx, len(levels) - 1)

            char = levels[idx]

            # -----------------------
            # ABOVE ZERO → worsening loss
            # -----------------------
            if v > 0:
                # red oscillator above baseline
                rendered.append((char, "bold red"))

            # -----------------------
            # BELOW ZERO → improving loss
            # -----------------------
            elif v < 0:
                rendered.append((char, "bold green"))

            else:
                rendered.append(("─", "white"))

        return rendered

    def render(
        self,
        epoch,
        epochs,
        train_loss,
        val_loss,
        lr,
        patience,
        best_val,
    ):

        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["lr"].append(lr)

        # -----------------------
        # Compact table (no expansion)
        # -----------------------
        table = Table(
            box=SIMPLE,
            show_header=False,
            expand=False,
            padding=(0, 1),
        )

        table.add_column("Metric")
        table.add_column("Value", justify="right")

        table.add_row("Train", f"{train_loss:.3e}")
        table.add_row("Val", f"{val_loss:.3e}")
        table.add_row("Best", f"{best_val:.3e}")
        table.add_row("LR", f"{lr:.2e}")
        table.add_row("Patience", str(patience))

        # -----------------------
        # Sparkline
        # -----------------------
        trend = self._sparkline_delta(self.history["val_loss"])

        trend_text = Text()
        trend_text.append("Val Δ Trend: ", style="bold")

        for char, style in trend:
            trend_text.append(char, style=style)

        # -----------------------
        # SINGLE COMPACT PANEL (IMPORTANT FIX)
        # -----------------------
        content = Table.grid(padding=1)
        content.add_row(table)
        content.add_row(trend_text)

        return Panel(
            content,
            title=f"Epoch {epoch+1}/{epochs}",
            expand=False,
            padding=(0, 1),
        )