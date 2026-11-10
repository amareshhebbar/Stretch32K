# Stretch32K

Take `TinyLlama/TinyLlama_v1.1`, a 1.1B model trained with a 2,048 token window, and stretch it to read 16k tokens (small config) or 32k tokens (big config) on one GPU. Six models trained on the same PG19 data with different context extension methods, compared on one table against the base model and the popular training free tricks.

Every model is pushed to the Hugging Face Hub as bf16 safetensors with its RoPE settings in `config.json`, added to the Stretch32K collection, downloaded back from the Hub and tested before the next model starts. No external APIs are used.

## What you get

<table>
<tr><th>Model</th><th>Positions</th><th>Training</th></tr>
<tr><td>PI</td><td>Position Interpolation: positions divided by the stretch factor</td><td>Full finetune on long PG19 chunks</td></tr>
<tr><td>NTK</td><td>NTK aware scaling: RoPE base frequency raised</td><td>Full finetune</td></tr>
<tr><td>YARN</td><td>YaRN: per frequency band interpolation plus attention temperature</td><td>Full finetune</td></tr>
<tr><td>LONGLORA</td><td>Position Interpolation</td><td>Shifted sparse attention, LoRA on attention, trainable embeddings and norms</td></tr>
<tr><td>YARN_LONGLORA</td><td>YaRN</td><td>Same LongLoRA recipe</td></tr>
<tr><td>SFT</td><td>Inherited from the best of the five</td><td>Full finetune on LongAlpaca long document instructions</td></tr>
</table>

Baseline rows on every card: the base model with no scaling, and the base model with PI, NTK, YaRN and dynamic NTK applied at load time with no training.

Tests: perplexity on PG19 test books at every length, passkey retrieval, a needle in a haystack heatmap over lengths and depths, a RULER subset (multiple keys, multiple values, variable tracking), and answer loss on held out LongAlpaca questions. The big config also runs HellaSwag and ARC Easy to show short text ability survived.

## Step 1: one time setup on your machine

1. Create an empty GitHub repo named `Stretch32K` (no README).
2. Unzip, then push:

```bash
unzip Stretch32K.zip && cd Stretch32K
git init
git branch -M main
git add .
git commit -m "Stretch32K pipeline"
git remote add origin https://github.com/amareshhebbar/Stretch32K.git
git push -u origin main
```

3. Create a Hugging Face token with Write access at huggingface.co/settings/tokens.

## Step 2: start the pod

* GPU: one 48 GB card. L40S is the best fit; A6000, A100 and H100 also work.
* Volume: at least 80 GB mounted at `/workspace` (network volume, so a restart keeps your progress).
* Template: any PyTorch template with CUDA 12. Python 3.10 to 3.12 is used if present, otherwise setup installs 3.11.

## Step 3: install and check

```bash
cd /workspace
git clone https://github.com/amareshhebbar/Stretch32K.git && cd Stretch32K
bash setup.sh
export HF_TOKEN=hf_yourtokenhere
export HF_HOME=/workspace/hf_cache
source .venv/bin/activate
python run.py check
```

Do not export `SX_WORK`. Each config sets its own run folder and `SX_WORK` would override it.

`check` runs six parts and prints PASS or FAIL for each. Total time is about 10 minutes.

<table>
<tr><th>Check</th><th>What it proves</th></tr>
<tr><td>sys</td><td>Libraries import, CUDA and bf16 work, the FlashAttention kernel runs causal attention at your training length, disk is enough</td></tr>
<tr><td>hub</td><td>Token has write access: creates a private repo and a collection, then deletes both</td></tr>
<tr><td>data</td><td>Base model and tokenizer load, PG19 train and test stream, LongAlpaca loads with the expected columns</td></tr>
<tr><td>rope</td><td>All five RoPE settings load, score perplexity at short and target length, and run passkey generation</td></tr>
<tr><td>s2</td><td>Shifted sparse attention matches full attention with one group and leaks nothing from future tokens</td></tr>
<tr><td>train</td><td>One real full finetune step and one LongLoRA step at your full sequence length fit in memory; prints tokens per second, peak memory and an estimate of total training hours; LoRA merge and save keep the RoPE settings</td></tr>
</table>

Run one part alone with `python run.py check train`. Read the time estimate from the train check before committing money to the full run.

## Step 4: run the pipeline

```bash
tmux new -s sx
source .venv/bin/activate
export HF_TOKEN=hf_yourtokenhere
export HF_HOME=/workspace/hf_cache
python run.py pipeline
```

Detach with `Ctrl+b` then `d`. Reattach with `tmux attach -t sx`. Watch logs from another terminal with:

```bash
tail -f /workspace/stretch32k_runs/logs/*.log
```

Steps, in order:

1. `factory`: tokenizes PG19 train books into packed chunks, caches PG19 test books for evaluation, prepares LongAlpaca, pushes everything as one dataset and checks row counts from the Hub.
2. `baseline`: evaluates the base model with no scaling and with the four training free tricks.
3. `baseline_short`: HellaSwag and ARC Easy for the base model (big config only).
4. For PI, NTK, YARN, LONGLORA, YARN_LONGLORA and then SFT:
   1. `train`
   2. `push`: weights, tokenizer, training logs, model card, added to the collection
   3. `verify`: deletes the local copy, downloads from the Hub, checks the RoPE settings, runs a quick perplexity and passkey test, and stops the pipeline if the model is broken
   4. `evalgen`: the full test suite plus the needle heatmap
   5. `shorteval`: short text benchmarks (big config only)
   6. `publish`: updates the card with the results table and charts
5. `report`: final charts and cards for every model.

If the pod stops, run `python run.py pipeline` again. Finished steps are skipped.

## Two configs

<table>
<tr><th></th><th>configs/config.yaml (default)</th><th>configs/config_big_budget.yaml</th></tr>
<tr><td>Stretch</td><td>2k to 16k (8x)</td><td>2k to 32k (16x)</td></tr>
<tr><td>Training per model</td><td>150 steps x 4 sequences x 16k tokens (about 10M tokens)</td><td>400 steps x 4 sequences x 32k tokens (about 52M tokens)</td></tr>
<tr><td>SFT examples</td><td>600</td><td>4,000</td></tr>
<tr><td>Eval lengths</td><td>1k to 16k</td><td>1k to 32k</td></tr>
<tr><td>Short text benchmarks</td><td>off</td><td>HellaSwag, ARC Easy</td></tr>
<tr><td>Run folder</td><td>/workspace/stretch32k_runs</td><td>/workspace/stretch32k_runs_big</td></tr>
<tr><td>Rough time on an L40S</td><td>about 4 to 6 hours</td><td>about 22 to 30 hours</td></tr>
</table>

The times are estimates; the train check prints a real one for your GPU. To run the big config, prefix every command:

```bash
SX_CONFIG=configs/config_big_budget.yaml python run.py check
SX_CONFIG=configs/config_big_budget.yaml python run.py pipeline
```

Both configs push to the same Hub repos, so a big run upgrades the small models in place.

## Useful commands

```bash
python run.py status
python run.py steps
python run.py run yarn:verify
python run.py pipeline from=longlora:train to=longlora:publish
python run.py reset yarn:train
python run.py reset all
```

## Saving money

* Stop the pod as soon as `pipeline complete` prints. An idle pod still bills.
* A spot or interruptible pod works well because the pipeline resumes.
* If budget runs out midway, every finished model is already on the Hub with its card. Run the rest later with `python run.py pipeline`.
* To trim further, lower `train.steps` in the config. 100 steps still teaches the new positions; retrieval gets weaker.

## If something fails

* Out of memory in the train check: lower `train.seq_len` and `target_len` together (for example 12288), or lower `logits_chunk` to 512.
* `token belongs to X but config hf_user is Y`: run `export HF_USER=X`.
* A verify failure means the uploaded model is broken (perplexity exploded). Check `/workspace/stretch32k_runs/results/<model>_verify.json`, then rerun that model with `python run.py reset <model>:train` and `python run.py pipeline`.
* Any other failure: the full log is in `/workspace/stretch32k_runs/logs/<step>.log`.

## Hub names

<table>
<tr><td>Dataset</td><td>AmareshHebbar/Stretch32K_Data_TinyLlama_1.1B</td></tr>
<tr><td>Models</td><td>AmareshHebbar/Stretch32K_TinyLlama_1.1B_PI, _NTK, _YARN, _LONGLORA, _YARN_LONGLORA, _SFT</td></tr>
<tr><td>Collection</td><td>Stretch32K</td></tr>
</table>

## How it works inside

* Memory: PyTorch SDPA FlashAttention kernels (no separate flash attention install), gradient checkpointing, fp32 master weights with bf16 autocast, the loss computed over the vocabulary in chunks so the full logits tensor never exists, and packed sequences with no padding.
* Shifted sparse attention: during LongLoRA training, half the heads attend inside fixed groups and half inside groups shifted by half a group. The shifted heads use separate short groups at the start and end instead of wrapping around, so no token can see the future. Inference always uses normal full attention.
* RoPE settings are written into each model's `config.json` (`rope_scaling`, `rope_theta`, `max_position_embeddings`), so the models load correctly anywhere.

## Layout

<table>
<tr><td>run.py</td><td>Entry point: check, pipeline, run, status, reset</td></tr>
<tr><td>sx/rope.py</td><td>PI, NTK, YaRN and dynamic NTK settings</td></tr>
<tr><td>sx/s2attn.py</td><td>Shifted sparse attention, registered as a transformers attention backend</td></tr>
<tr><td>sx/data.py</td><td>PG19 streaming and packing, eval books, LongAlpaca, dataset push</td></tr>
<tr><td>sx/modeling.py</td><td>Chunked loss and model loading</td></tr>
<tr><td>sx/train.py</td><td>Continued pretraining and SFT</td></tr>
<tr><td>sx/evaluate.py</td><td>Perplexity, passkey, needle grid, RULER subset, LongAlpaca loss, short benchmarks</td></tr>
<tr><td>sx/verify.py</td><td>Hub download and quality gate</td></tr>
<tr><td>sx/hub.py, sx/card.py, sx/report.py</td><td>Uploads, collection, model cards, heatmaps and curves</td></tr>
<tr><td>sx/check.py</td><td>Environment checks</td></tr>
</table>
