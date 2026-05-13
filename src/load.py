import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.config import DEFAULT_PRECISION, DRAFT_DEVICE_MAP, DRAFT_MODEL_ID, TARGET_DEVICE_MAP, TARGET_MODEL_ID

console = Console()

def _quant_config(precision: str):
    if precision == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "4bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",)
    return None

def _torch_dtype(precision: str):
    if precision in ("bf16",):
        return torch.bfloat16
    if precision in ("8bit", "4bit"):
        return torch.bfloat16
    return torch.float32

def load_tokenizer(model_id: str = TARGET_MODEL_ID):
    console.print(f"[cyan]Loading tokenizer from[/cyan] [bold]{model_id}[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_id: str, precision: str, device_map: str, label: str):
    quant_cfg = _quant_config(precision)
    dtype = _torch_dtype(precision)
    device_map_ = device_map if precision != "cpu" else {"": "cpu"}

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console,) as progress:
        progress.add_task(f"Loading {label} weights…", total=None)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map_, quantization_config=quant_cfg,)
    model.eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    console.print(f"{label} loaded — [green]{param_count:.1f}B params[/green]")
    return model

def load_both(precision: str = DEFAULT_PRECISION):
    console.rule("[bold blue]Speculative Decoding — Model Loader")
    tokenizer = load_tokenizer(TARGET_MODEL_ID)
    draft_model = load_model(DRAFT_MODEL_ID, precision, DRAFT_DEVICE_MAP, "Draft (1B)",)
    target_model = load_model(TARGET_MODEL_ID, precision, TARGET_DEVICE_MAP, "Target (12B)",)
    console.rule("[bold green]Both models ready")
    _print_vram_usage()
    return tokenizer, draft_model, target_model

def _print_vram_usage():
    if not torch.cuda.is_available():
        console.print("[yellow]No CUDA device detected — running on CPU[/yellow]")
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        console.print(f"  GPU {i}: [magenta]{alloc:.1f}GB[/magenta] / {total:.1f}GB allocated")
