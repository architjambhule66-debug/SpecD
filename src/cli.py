import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from src.config import DEFAULT_DRAFT_K, DEFAULT_MAX_NEW_TOKENS, DEFAULT_PRECISION, DEFAULT_TEMPERATURE, DRAFT_MODEL_ID, PRECISION_CHOICES, TARGET_MODEL_ID
from src.generate import speculative_generate
from src.load import load_both

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Speculative Decoding with Gemma 3 (1B draft + 12B target)")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--k", type=int, default=DEFAULT_DRAFT_K, help="Draft tokens per cycle (default: 5)",)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--precision", type=str, default=DEFAULT_PRECISION, choices=PRECISION_CHOICES,)
    return parser.parse_args()


def main():
    args = parse_args()

    console.print(
        Panel.fit(
            "[bold blue]Speculative Decoding[/bold blue]\n"
            f"Draft: [cyan]{DRAFT_MODEL_ID}[/cyan]\n"
            f"Target: [cyan]{TARGET_MODEL_ID}[/cyan]\n"
            f"k={args.k} | temp={args.temperature} | precision={args.precision}",
            border_style="blue",
        )
    )

    tokenizer, draft_model, target_model = load_both(args.precision)

    if args.prompt:
        prompts = [args.prompt]
    else:
        console.print("\n[dim]Enter prompts interactively. Type 'quit' to exit.[/dim]\n")
        prompts = None

    while True:
        if prompts:
            prompt = prompts.pop(0)
        else:
            prompt = Prompt.ask("\n[bold green]Prompt")
            if prompt.lower() in ("quit", "exit", "q"):
                break

        console.print(f"\n[dim]Generating (k={args.k})...[/dim]\n")

        text, metrics = speculative_generate(tokenizer, draft_model, target_model, prompt, max_new_tokens=args.max_tokens, k=args.k, temperature=args.temperature, stream=True,)

        console.print(Panel(text, title="[bold]Output", border_style="green"))
        metrics.print_summary()

        if prompts is not None and len(prompts) == 0:
            break


if __name__ == "__main__":
    main()
