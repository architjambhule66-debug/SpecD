import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rich.console import Console
from rich import box
from rich.table import Table

from src.config import BENCHMARK_DEFAULT_DRAFT_K, BENCHMARK_DEFAULT_TEMPERATURE, BENCHMARK_PROMPTS, DEFAULT_MAX_NEW_TOKENS, DEFAULT_PRECISION, DEFAULT_TOP_P, DRAFT_MODEL_ID, PRECISION_CHOICES, TARGET_MODEL_ID,
from src.load import load_both
from src.generate import speculative_generate, vanilla_generate
from src.tracker import BenchmarkResult

console = Console()


def run_benchmark(precision: str = DEFAULT_PRECISION,k: int = BENCHMARK_DEFAULT_DRAFT_K,temperature: float = BENCHMARK_DEFAULT_TEMPERATURE,max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,prompt_limit: int | None = None,):
    prompts = (
        BENCHMARK_PROMPTS[:prompt_limit]
        if prompt_limit is not None
        else BENCHMARK_PROMPTS
    )

    console.rule("[bold magenta]Speculative Decoding Benchmark")
    console.print(f"  Draft  : [cyan]{DRAFT_MODEL_ID}[/cyan]")
    console.print(f"  Target : [cyan]{TARGET_MODEL_ID}[/cyan]")
    console.print(f"  k      : [yellow]{k}[/yellow] draft tokens per cycle")
    console.print(f"  Temp   : [yellow]{temperature}[/yellow]")
    console.print(f"  Tokens : [yellow]{max_new_tokens}[/yellow]")
    console.print(f"  Precision: [yellow]{precision}[/yellow]\n")

    # Load models once, reuse across all prompts
    tokenizer, draft_model, target_model = load_both(precision)

    results: list[BenchmarkResult] = []

    for i, prompt in enumerate(prompts):
        console.rule(f"[bold]Prompt {i + 1}/{len(prompts)}")
        console.print(f"[dim]{prompt}[/dim]\n")

        # ── Vanilla baseline ────────────────────────────────────────────────
        console.print("[yellow]Running vanilla generation...[/yellow]")
        _, v_metrics = vanilla_generate(
            tokenizer,
            target_model,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
        )
        v_metrics.print_summary("Vanilla")
        vanilla_tps = v_metrics.tokens_per_second

        # ── Speculative decoding ────────────────────────────────────────────
        console.print("\n[green]Running speculative decoding...[/green]")
        _, s_metrics = speculative_generate(
            tokenizer,
            draft_model,
            target_model,
            prompt,
            max_new_tokens=max_new_tokens,
            k=k,
            temperature=temperature,
            top_p=DEFAULT_TOP_P,
            stream=False,
        )
        s_metrics.print_summary("Speculative")
        spec_tps = s_metrics.tokens_per_second

        results.append(
            BenchmarkResult(
                prompt=prompt,
                vanilla_tps=vanilla_tps,
                speculative_tps=spec_tps,
                acceptance_rate=s_metrics.mean_acceptance_rate,
                speedup=spec_tps / max(vanilla_tps, 0.01),
            )
        )

    # ── Summary table ───────────────────────────────────────────────────────
    console.rule("[bold cyan]Final Results")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Prompt", style="dim", max_width=35)
    table.add_column("Vanilla tok/s", justify="right")
    table.add_column("Spec tok/s", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold")
    table.add_column("Accept Rate", justify="right")

    for r in results:
        color = "green" if r.speedup > 1.5 else "yellow" if r.speedup > 1.0 else "red"
        table.add_row(
            r.prompt[:35],
            f"{r.vanilla_tps:.1f}",
            f"{r.speculative_tps:.1f}",
            f"[{color}]{r.speedup:.2f}x[/{color}]",
            f"{r.acceptance_rate:.1%}",
        )

    console.print(table)

    avg_speedup = sum(r.speedup for r in results) / len(results)
    avg_accept = sum(r.acceptance_rate for r in results) / len(results)
    console.print(f"\n[bold]Average speedup  : [green]{avg_speedup:.2f}x[/green]")
    console.print(f"[bold]Average accept % : [green]{avg_accept:.1%}[/green]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=PRECISION_CHOICES)
    parser.add_argument("--k", type=int, default=BENCHMARK_DEFAULT_DRAFT_K)
    parser.add_argument("--temperature", type=float, default=BENCHMARK_DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--prompt-limit", type=int, default=None)
    args = parser.parse_args()
    run_benchmark(precision=args.precision,k=args.k,temperature=args.temperature,max_new_tokens=args.max_tokens,prompt_limit=args.prompt_limit,)
