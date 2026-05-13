import time
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class GenerationMetrics:
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    total_tokens_generated: int = 0
    prompt_tokens: int = 0
    total_cycles: int = 0
    total_accepted: int = 0
    total_drafted: int = 0
    acceptance_rates: list  = field(default_factory=list)

    def record_cycle(self, num_accepted: int, k: int, tokens_produced: int):
        self.total_cycles += 1
        self.total_accepted += num_accepted
        self.total_drafted += k
        self.total_tokens_generated += tokens_produced
        self.acceptance_rates.append(num_accepted / k)

    def finish(self):
        self.end_time = time.time()

    @property
    def elapsed_seconds(self) -> float:
        t = self.end_time if self.end_time else time.time()
        return t - self.start_time

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.total_tokens_generated / self.elapsed_seconds

    @property
    def mean_acceptance_rate(self) -> float:
        if not self.acceptance_rates:
            return 0.0
        return sum(self.acceptance_rates) / len(self.acceptance_rates)

    @property
    def effective_speedup(self) -> float:
        if self.total_cycles == 0:
            return 1.0
        tokens_per_cycle = self.total_tokens_generated / self.total_cycles
        return tokens_per_cycle

    def print_summary(self, label: str = "Speculative Decoding"):
        console.rule(f"[bold cyan]{label} — Performance Summary")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value",  style="green")
        table.add_row("Total tokens generated", str(self.total_tokens_generated))
        table.add_row("Elapsed time",           f"{self.elapsed_seconds:.2f}s")
        table.add_row("Tokens / second",        f"[bold green]{self.tokens_per_second:.1f}[/bold green]")
        table.add_row("Draft-verify cycles",    str(self.total_cycles))
        table.add_row("Draft tokens proposed",  str(self.total_drafted))
        table.add_row("Draft tokens accepted",  str(self.total_accepted))
        table.add_row("Mean acceptance rate", f"[{'green' if self.mean_acceptance_rate > 0.7 else 'yellow'}]" f"{self.mean_acceptance_rate:.1%}[/]")
        table.add_row("Effective tokens/cycle", f"{self.effective_speedup:.2f}x  (vs 1.0x vanilla)")
        console.print(table)


@dataclass
class BenchmarkResult:
    prompt:             str
    vanilla_tps:        float
    speculative_tps:    float
    acceptance_rate:    float
    speedup:            float

    def __str__(self):
        return f"""
            Prompt: {self.prompt[:50]}...\n
            Vanilla:     {self.vanilla_tps:.1f} tok/s\n
            Speculative: {self.speculative_tps:.1f} tok/s\n
            Speedup:     {self.speedup:.2f}x\n
            Accept rate: {self.acceptance_rate:.1%}
        """