#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Fresh-clone Linux launcher for storage-safe grokking-geometry runs.

Usage:
  bash run_fresh_linux.sh --mode quick
  bash run_fresh_linux.sh --mode pilot
  bash run_fresh_linux.sh --mode full
  bash run_fresh_linux.sh --mode ablation

Options:
  --mode quick|pilot|full|ablation   Run mode (default: pilot)
  --cpu                              Install/use CPU PyTorch and force device=cpu
  --skip-install                     Reuse the current Python environment without pip installs
  --system-python                    Use the current Python environment instead of creating .venv
  --skip-torch-install               Do not reinstall torch/torchvision/torchaudio
  --run-root NAME                    Output run root
  --dataset NAME|all                 Dataset chunk for full/ablation modes
  --layers LIST                      Comma-separated layer grid
  --widths LIST                      Comma-separated width grid
  --seeds LIST                       Comma-separated seed grid
  --weight-decays LIST               Comma-separated grid for ablation mode
  --train-fractions LIST             Comma-separated grid for ablation mode
  --cuda-index-url URL               PyTorch CUDA wheel index
  --resume                           Skip runs with signals.csv and restart incomplete folders
  --keep-checkpoints                 Keep checkpoints after analysis
  --max-run-gb GB                    Per-run storage cap (default: 9.5)
  --extra-overrides "..."            Extra Hydra overrides
EOF
}

mode="pilot"
cpu=0
skip_install=0
system_python=0
skip_torch_install=0
run_root=""
dataset="all"
layers_override=""
widths_override=""
seeds_override=""
cuda_index_url="https://download.pytorch.org/whl/cu128"
weight_decays="0.0,0.1,1.0"
train_fractions="0.25,0.5,1.0"
resume=0
keep_checkpoints=0
max_run_gb="9.5"
extra_overrides=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:?missing value for --mode}"
      shift 2
      ;;
    --cpu)
      cpu=1
      shift
      ;;
    --skip-install)
      skip_install=1
      system_python=1
      skip_torch_install=1
      shift
      ;;
    --system-python)
      system_python=1
      shift
      ;;
    --skip-torch-install)
      skip_torch_install=1
      shift
      ;;
    --run-root)
      run_root="${2:?missing value for --run-root}"
      shift 2
      ;;
    --dataset)
      dataset="${2:?missing value for --dataset}"
      shift 2
      ;;
    --layers)
      layers_override="${2:?missing value for --layers}"
      shift 2
      ;;
    --widths)
      widths_override="${2:?missing value for --widths}"
      shift 2
      ;;
    --seeds)
      seeds_override="${2:?missing value for --seeds}"
      shift 2
      ;;
    --weight-decays)
      weight_decays="${2:?missing value for --weight-decays}"
      shift 2
      ;;
    --train-fractions)
      train_fractions="${2:?missing value for --train-fractions}"
      shift 2
      ;;
    --cuda-index-url)
      cuda_index_url="${2:?missing value for --cuda-index-url}"
      shift 2
      ;;
    --resume)
      resume=1
      shift
      ;;
    --keep-checkpoints)
      keep_checkpoints=1
      shift
      ;;
    --max-run-gb)
      max_run_gb="${2:?missing value for --max-run-gb}"
      shift 2
      ;;
    --extra-overrides)
      extra_overrides="${2:?missing value for --extra-overrides}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

case "$mode" in
  quick|pilot|full|ablation) ;;
  *)
    echo "--mode must be one of quick, pilot, full, ablation" >&2
    exit 2
    ;;
esac

if [[ ! -f run.py ]]; then
  echo "Run this script from the repository root, next to run.py." >&2
  exit 1
fi

if [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d /kaggle ]]; then
  if [[ "$system_python" -eq 0 ]]; then
    echo "Detected Kaggle; using the current Python environment and existing PyTorch."
    system_python=1
    skip_torch_install=1
  fi
fi

if [[ "$system_python" -eq 1 ]]; then
  python_bin="${PYTHON:-python3}"
else
  if [[ ! -x .venv/bin/python ]]; then
    echo "Creating .venv..."
    python3 -m venv .venv
  fi
  python_bin=".venv/bin/python"
fi

install_requirements() {
  if [[ "$skip_torch_install" -eq 1 ]]; then
    local filtered_req
    filtered_req="$(mktemp)"
    grep -Ev '^[[:space:]]*(torch|torchvision|torchaudio)([<>=[:space:]]|$)' requirements.txt > "$filtered_req"
    "$python_bin" -m pip install -r "$filtered_req"
    rm -f "$filtered_req"
  else
    "$python_bin" -m pip install -r requirements.txt
  fi
}

if [[ "$skip_install" -eq 0 ]]; then
  echo "Installing Python dependencies..."
  if [[ "$system_python" -eq 0 ]]; then
    "$python_bin" -m pip install --upgrade pip
  fi
  if [[ "$skip_torch_install" -eq 0 ]]; then
    if [[ "$cpu" -eq 1 ]]; then
      "$python_bin" -m pip install torch torchvision torchaudio
    else
      "$python_bin" -m pip install torch torchvision torchaudio --index-url "$cuda_index_url"
    fi
  else
    "$python_bin" - <<'PY'
import torch
print(f"Using existing PyTorch {torch.__version__}; cuda_available={torch.cuda.is_available()}")
PY
  fi
  install_requirements
fi

echo "Checking Python dependencies..."
"$python_bin" - <<'PY'
import importlib.util

modules = [
    ("torch", "torch"),
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("wandb", "wandb"),
    ("tqdm", "tqdm"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("ruptures", "ruptures"),
    ("statsmodels", "statsmodels"),
    ("seaborn", "seaborn"),
]
missing = [package for module, package in modules if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(
        "Missing Python packages: "
        + ", ".join(missing)
        + ". Re-run without --skip-install, or install them in the active environment."
    )
PY

echo "Preparing datasets..."
"$python_bin" data/modular_addition/prepare.py
"$python_bin" data/modular_subtraction/prepare.py
"$python_bin" data/modular_multiplication/prepare.py
"$python_bin" data/symmetric_group/prepare.py
"$python_bin" data/permutation_composition/prepare.py

if [[ -z "$run_root" ]]; then
  run_root="runs_${mode}_$(date +%Y%m%d_%H%M%S)"
fi

extra="analysis.interventions.enabled=true"
if [[ "$cpu" -eq 1 ]]; then
  extra="$extra device=cpu"
fi
if [[ -n "$extra_overrides" ]]; then
  extra="$extra $extra_overrides"
fi

storage_policy=(
  "run_root=$run_root"
  "save_every=200"
  "eval_interval=200"
  "analysis.stride=1"
  "storage.save_optimizer=false"
  "storage.save_rng=false"
  "storage.max_checkpoints=60"
  "storage.max_run_gb=$max_run_gb"
  "storage.keep_last_checkpoints=0"
  "wandb.mode=disabled"
)

if [[ "$keep_checkpoints" -eq 1 ]]; then
  storage_policy+=(
    "storage.keep_checkpoints_after_analysis=true"
    "storage.keep_last_checkpoints=1"
  )
fi

read_csv_grid() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
  for i in "${!out_ref[@]}"; do
    out_ref[$i]="${out_ref[$i]//[[:space:]]/}"
  done
}

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  echo "$value"
}

run_one() {
  local ds="$1"
  local layers="$2"
  local width="$3"
  local seed="$4"
  local wd="$5"
  local tf="$6"
  local nhead=$(( width / 32 ))
  if [[ "$nhead" -lt 1 ]]; then
    nhead=1
  fi

  local exp_name="${ds}_L${layers}_d${width}_wd$(float_tag "$wd")_tf$(float_tag "$tf")_seed${seed}"
  local out_dir="${run_root}/${exp_name}"
  if [[ "$resume" -eq 1 ]]; then
    if [[ -f "${out_dir}/signals.csv" ]]; then
      echo "[$count/$total] SKIP $exp_name (signals.csv found)"
      return
    elif [[ -d "$out_dir" ]]; then
      find "$out_dir" -maxdepth 1 -type f \( -name 'ckpt_*.pt' -o -name 'signals.png' \) -delete
    fi
  fi

  echo "[$count/$total] RUN $exp_name"
  local cmd=(
    "$python_bin" run.py
    "${storage_policy[@]}"
    dataset="$ds"
    model.n_layer="$layers"
    model.n_embd="$width"
    model.n_head="$nhead"
    seed="$seed"
    weight_decay="$wd"
    train_fraction="$tf"
  )

  if [[ "$mode" == "quick" ]]; then
    cmd+=(
      max_iters=500
      analysis.hutchinson_samples=1
      analysis.power_iters=5
      analysis.num_examples=64
      analysis.sinkhorn_iters=10
    )
  elif [[ "$mode" == "pilot" ]]; then
    cmd+=(
      max_iters=800
      eval_interval=400
      save_every=400
      analysis.hutchinson_samples=1
      analysis.power_iters=5
      analysis.num_examples=64
      analysis.sinkhorn_iters=10
    )
  elif [[ "$ds" == "symmetric_group" ]]; then
    cmd+=(max_iters=20000 analysis.stride=1)
  else
    cmd+=(max_iters=10000 analysis.stride=1)
  fi

  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    local extra_args=( $extra )
    cmd+=("${extra_args[@]}")
  fi

  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  local start end elapsed
  start=$(date +%s)
  if ! "${cmd[@]}"; then
    end=$(date +%s)
    elapsed=$(( end - start ))
    echo "  FAILED: $exp_name" >&2
    printf '  Failed after %.1f min\n\n' "$("$python_bin" - <<PY
print($elapsed / 60)
PY
)" >&2
    return 1
  fi
  end=$(date +%s)
  elapsed=$(( end - start ))
  printf '  Completed in %.1f min\n\n' "$("$python_bin" - <<PY
print($elapsed / 60)
PY
)"
}

if [[ "$mode" == "quick" ]]; then
  datasets=(modular_addition)
  layers_grid=(1)
  widths=(128)
  seeds=(42)
  wd_grid=(1.0)
  tf_grid=(1.0)
elif [[ "$mode" == "pilot" ]]; then
  datasets=(modular_addition)
  layers_grid=(1)
  widths=(64 128)
  seeds=(0)
  wd_grid=(0.0 1.0)
  tf_grid=(0.25 1.0)
elif [[ "$mode" == "ablation" ]]; then
  if [[ "$dataset" == "all" ]]; then
    datasets=(modular_addition modular_multiplication symmetric_group)
  else
    datasets=("$dataset")
  fi
  layers_grid=(1 2 4 6)
  widths=(64 128 256 512)
  seeds=(0 1 2 3 4)
  read_csv_grid "$weight_decays" wd_grid
  read_csv_grid "$train_fractions" tf_grid
else
  if [[ "$dataset" == "all" ]]; then
    datasets=(modular_addition modular_multiplication symmetric_group)
  else
    datasets=("$dataset")
  fi
  layers_grid=(1 2 4 6)
  widths=(64 128 256 512)
  seeds=(0 1 2 3 4)
  wd_grid=(1.0)
  tf_grid=(1.0)
fi

if [[ -n "$layers_override" ]]; then
  read_csv_grid "$layers_override" layers_grid
fi
if [[ -n "$widths_override" ]]; then
  read_csv_grid "$widths_override" widths
fi
if [[ -n "$seeds_override" ]]; then
  read_csv_grid "$seeds_override" seeds
fi

total=$(( ${#datasets[@]} * ${#layers_grid[@]} * ${#widths[@]} * ${#seeds[@]} * ${#wd_grid[@]} * ${#tf_grid[@]} ))
count=0

echo "============================================"
echo "  SWEEP: $total configurations"
echo "============================================"
echo "Mode     : $mode"
echo "Datasets : ${datasets[*]}"
echo "Layers   : ${layers_grid[*]}"
echo "Widths   : ${widths[*]}"
echo "Seeds    : ${seeds[*]}"
echo "WD       : ${wd_grid[*]}"
echo "Train Fr.: ${tf_grid[*]}"
echo "Run root : $run_root"
echo ""

for ds in "${datasets[@]}"; do
  for layers in "${layers_grid[@]}"; do
    for width in "${widths[@]}"; do
      for seed in "${seeds[@]}"; do
        for wd in "${wd_grid[@]}"; do
          for tf in "${tf_grid[@]}"; do
            count=$(( count + 1 ))
            run_one "$ds" "$layers" "$width" "$seed" "$wd" "$tf"
          done
        done
      done
    done
  done
done

echo "============================================"
echo "  SWEEP COMPLETE: $count / $total runs"
echo "  Aggregating results into results/$run_root"
echo "============================================"

"$python_bin" summarize_runs.py --runs-dir "$run_root" --output-dir "results/$run_root"

echo "Done."
echo "Run root: $run_root"
echo "Aggregates: results/$run_root"
