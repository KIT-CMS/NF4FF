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
        trend = self._sparkline(self.history["val_loss"])

        trend_text = Text()
        trend_text.append("Val Trend: ", style="bold")
        trend_text.append(trend)

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