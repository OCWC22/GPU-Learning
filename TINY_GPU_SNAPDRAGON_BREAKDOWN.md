# Complete Guide: From tiny-gpu to Snapdragon 8 Elite to NVIDIA B200

## How GPUs Work From First Principles — Using 12 Files of SystemVerilog to Understand Billion-Transistor Chips

> **Source material**: [tiny-gpu](https://github.com/AmaadMartin/tiny-gpu) by Adam Majmudar — a minimal GPU implementation in SystemVerilog designed as an educational project.
>
> **Target hardware**: Qualcomm Snapdragon 8 Elite (SM8750) with Adreno 830 GPU and Hexagon NPU, Samsung Galaxy S25+, NVIDIA B200 (Blackwell).

---

## Table of Contents

- [Part I: Foundations](#part-i-foundations)
  - [Chapter 0: What Even Is Hardware?](#chapter-0-what-even-is-hardware)
  - [Chapter 1: The One Law That Rules Everything](#chapter-1-the-one-law-that-rules-everything)
  - [Chapter 2: The Roofline Model](#chapter-2-the-roofline-model)
  - [Chapter 3: Why GPUs Exist](#chapter-3-why-gpus-exist)
  - [Chapter 4: What Is Verilog / SystemVerilog?](#chapter-4-what-is-verilog--systemverilog)
  - [Chapter 5: From Verilog to Real Chips](#chapter-5-from-verilog-to-real-chips)
- [Part II: tiny-gpu Architecture](#part-ii-tiny-gpu-architecture)
  - [Chapter 6: The Programming Model — SIMD](#chapter-6-the-programming-model--simd)
  - [Chapter 7: Top-Level Architecture — gpu.sv](#chapter-7-top-level-architecture--gpusv)
  - [Chapter 8: The Device Control Register — dcr.sv](#chapter-8-the-device-control-register--dcrsv)
  - [Chapter 9: The Dispatcher — dispatch.sv](#chapter-9-the-dispatcher--dispatchsv)
  - [Chapter 10: The Compute Core — core.sv](#chapter-10-the-compute-core--coresv)
  - [Chapter 11: The Scheduler — scheduler.sv](#chapter-11-the-scheduler--schedulersv)
  - [Chapter 12: The Instruction Fetcher — fetcher.sv](#chapter-12-the-instruction-fetcher--fetchersv)
  - [Chapter 13: The Decoder — decoder.sv](#chapter-13-the-decoder--decodersv)
  - [Chapter 14: The Register File — registers.sv](#chapter-14-the-register-file--registerssv)
  - [Chapter 15: The ALU — alu.sv](#chapter-15-the-alu--alusv)
  - [Chapter 16: The Program Counter — pc.sv](#chapter-16-the-program-counter--pcsv)
  - [Chapter 17: The Load-Store Unit — lsu.sv](#chapter-17-the-load-store-unit--lsusv)
  - [Chapter 18: The Memory Controller — controller.sv](#chapter-18-the-memory-controller--controllersv)
  - [Chapter 19: The ISA](#chapter-19-the-isa)
- [Part II.5: Code Walkthrough — Line-by-Line](#part-ii5-code-walkthrough--line-by-line)
  - [Chapter 19.5: gpu.sv — Two Memory Controllers](#chapter-195-gpusv--two-memory-controllers)
  - [Chapter 19.6: dispatch.sv — Block Distribution](#chapter-196-dispatchsv--block-distribution)
  - [Chapter 19.7: scheduler.sv — The 7-State FSM](#chapter-197-schedulersv--the-7-state-fsm)
  - [Chapter 19.8: fetcher.sv — Async Instruction Fetch](#chapter-198-fetchersv--async-instruction-fetch)
  - [Chapter 19.9: decoder.sv — Instruction Decoding](#chapter-199-decodersv--instruction-decoding)
  - [Chapter 19.10: registers.sv — SIMD Identity](#chapter-1910-registerssv--simd-identity)
  - [Chapter 19.11: alu.sv — Compute is Free](#chapter-1911-alusv--compute-is-free)
  - [Chapter 19.12: lsu.sv — Async Memory](#chapter-1912-lsusv--async-memory)
  - [Chapter 19.13: controller.sv — Priority Arbiter](#chapter-1913-controllersv--priority-arbiter)
  - [Chapter 19.14: Memory Coalescing Analysis](#chapter-1914-memory-coalescing-analysis)
  - [Chapter 19.15: Full Execution Trace](#chapter-1915-full-execution-trace)
  - [Chapter 19.16: Energy Breakdown](#chapter-1916-energy-breakdown)
- [Part III: Execution Trace — Step by Step](#part-iii-execution-trace--step-by-step)
  - [Chapter 20: Launching a Kernel](#chapter-20-launching-a-kernel)
  - [Chapter 21: Cycle-by-Cycle Execution of matadd](#chapter-21-cycle-by-cycle-execution-of-matadd)
- [Part IV: The Five Problems](#part-iv-the-five-problems)
  - [Chapter 22: Instruction Fetch Bottleneck](#chapter-22-instruction-fetch-bottleneck)
  - [Chapter 23: No Memory Coalescing](#chapter-23-no-memory-coalescing)
  - [Chapter 24: No Data Cache](#chapter-24-no-data-cache)
  - [Chapter 25: Core Stalls on Memory](#chapter-25-core-stalls-on-memory)
  - [Chapter 26: Unfair Memory Arbitration](#chapter-26-unfair-memory-arbitration)
- [Part V: Stride, Banking, and Conflicts](#part-v-stride-banking-and-conflicts)
  - [Chapter 27: How the Controller Arbitrates](#chapter-27-how-the-controller-arbitrates)
  - [Chapter 28: Stride Patterns](#chapter-28-stride-patterns)
  - [Chapter 29: Memory Banks](#chapter-29-memory-banks)
  - [Chapter 30: The Handshake Protocol](#chapter-30-the-handshake-protocol)
- [Part VI: Mapping to Snapdragon 8 Elite](#part-vi-mapping-to-snapdragon-8-elite)
  - [Chapter 31: Snapdragon SoC Overview](#chapter-31-snapdragon-soc-overview)
  - [Chapter 32: gpu.sv → Adreno 830](#chapter-32-gpusv--adreno-830)
  - [Chapter 33: dcr.sv → Command Processor](#chapter-33-dcrsv--command-processor)
  - [Chapter 34: dispatch.sv → Workgroup Scheduler](#chapter-34-dispatchsv--workgroup-scheduler)
  - [Chapter 35: core.sv → Compute Unit](#chapter-35-coresv--compute-unit)
  - [Chapter 36: scheduler.sv → Wave Scheduler](#chapter-36-schedulersv--wave-scheduler)
  - [Chapter 37: controller.sv → Memory Hierarchy](#chapter-37-controllersv--memory-hierarchy)
  - [Chapter 38: alu.sv → Shader ALUs](#chapter-38-alusv--shader-alus)
  - [Chapter 39: lsu.sv → Load/Store + Coalescing](#chapter-39-lsusv--loadstore--coalescing)
  - [Chapter 40: The Power Dimension](#chapter-40-the-power-dimension)
  - [Chapter 41: The Software Stack](#chapter-41-the-software-stack)
- [Part VII: Inference Side-by-Side](#part-vii-inference-side-by-side)
  - [Chapter 42: Setup Phase](#chapter-42-setup-phase)
  - [Chapter 43: Dispatch Phase](#chapter-43-dispatch-phase)
  - [Chapter 44: Computing the Thread Index](#chapter-44-computing-the-thread-index)
  - [Chapter 45: The First Memory Load](#chapter-45-the-first-memory-load)
  - [Chapter 46: The Second Memory Load — Cache Payoff](#chapter-46-the-second-memory-load--cache-payoff)
  - [Chapter 47: The Actual Computation](#chapter-47-the-actual-computation)
  - [Chapter 48: The Store](#chapter-48-the-store)
  - [Chapter 49: Kernel Complete](#chapter-49-kernel-complete)
  - [Chapter 50: Complete Timeline Comparison](#chapter-50-complete-timeline-comparison)
- [Part VIII: The Bandwidth Wall and Real Inference](#part-viii-the-bandwidth-wall-and-real-inference)
  - [Chapter 51: The Bandwidth Wall — Applied to Real Hardware](#chapter-51-the-bandwidth-wall--applied-to-real-hardware)
  - [Chapter 52: LLM Inference Mapped to tiny-gpu Problems](#chapter-52-llm-inference-mapped-to-tiny-gpu-problems)
  - [Chapter 53: What We Know vs What We Don't](#chapter-53-what-we-know-vs-what-we-dont)
  - [Chapter 53.5: Prefill vs Decode — Two Radically Different Workloads](#chapter-535-prefill-vs-decode--two-radically-different-workloads)
  - [Chapter 53.6: Quantization Effects on Bandwidth — The Full Math](#chapter-536-quantization-effects-on-bandwidth--the-full-math)
  - [Chapter 53.7: Systolic Arrays vs Dataflow — Two Memory Hierarchies](#chapter-537-systolic-arrays-vs-dataflow--two-memory-hierarchies)
  - [Chapter 53.8: Shared Memory and Micro Tiling — Keeping Data On-Chip](#chapter-538-shared-memory-and-micro-tiling--keeping-data-on-chip)
  - [Chapter 53.9: Prefetching and Lookahead — Hiding Latency with Data Movement](#chapter-539-prefetching-and-lookahead--hiding-latency-with-data-movement)
  - [Chapter 53.10: Batch Size, Wave Residency, and Register File Explosion](#chapter-5310-batch-size-wave-residency-and-register-file-explosion)
  - [Chapter 53.11: Sustained vs Burst Performance — The Thermal Ceiling](#chapter-5311-sustained-vs-burst-performance--the-thermal-ceiling)
- [Part IX: NVIDIA B200 — Same Physics, Different Constants](#part-ix-nvidia-b200--same-physics-different-constants)
  - [Chapter 54: B200 Architecture Overview](#chapter-54-b200-architecture-overview)
  - [Chapter 55: The Constants That Change Everything](#chapter-55-the-constants-that-change-everything)
- [Part X: Complete Reference](#part-x-complete-reference)
  - [Chapter 56: Snapdragon 8 Elite Full Architecture](#chapter-56-snapdragon-8-elite-full-architecture)
  - [Chapter 57: The Complete Mapping Table](#chapter-57-the-complete-mapping-table)
  - [Chapter 58: Learning Roadmap](#chapter-58-learning-roadmap)

---

# Part I: Foundations

## Chapter 0: What Even Is Hardware?

Everything in your phone or computer is built from **transistors** — tiny electrical switches that are either ON (1) or OFF (0). That's it. Every GPU, CPU, everything — just billions of switches.

```
Transistor = a switch

    ON  = 1 = electricity flows
    OFF = 0 = electricity doesn't flow
```

You combine switches to make **logic gates**:

```
AND gate:  both inputs ON  → output ON
OR gate:   either input ON → output ON
NOT gate:  flips the input

    A ──┐
        ├─ AND ── output (1 only if A=1 AND B=1)
    B ──┘
```

You combine logic gates to make **circuits** — adders, multiplexers, registers, memory.

You combine circuits to make **processors** — CPUs, GPUs.

That's the entire stack:

```
Transistors → Logic Gates → Circuits → Processors → Computers
```

### Electrical Engineering Layers

Within EE, there are layers:

```
Layer 1: Physics / Materials
  → Silicon, doping, how transistors physically work
  → "Why does a MOSFET switch?"

Layer 2: Circuit Design
  → Combining transistors into logic gates
  → Analog vs digital, voltage levels, timing
  → "How fast can this gate switch?"

Layer 3: Digital Design / RTL (Register Transfer Level)
  → Describing circuits in a hardware description language (Verilog / SystemVerilog)
  → THIS IS WHAT tiny-gpu IS
  → "What does the circuit DO, logically?"

Layer 4: Architecture
  → How you organize thousands of circuits into a processor
  → "How many cores? How does memory work? What's the ISA?"

Layer 5: Systems
  → How the processor talks to memory, display, network
  → "How does the GPU talk to the CPU? To RAM?"
```

**tiny-gpu lives at Layers 3 and 4.** It's written in SystemVerilog (Layer 3) and defines a GPU architecture (Layer 4).

---

## Chapter 1: The One Law That Rules Everything

Before we look at any code or any chip, there is one physical truth that dominates all inference hardware design:

**Moving data costs far more than computing on it.**

This isn't an opinion or a design choice. It's physics. The canonical reference is Horowitz's energy table (originally 45nm CMOS; absolute pJ values shrink at 3nm but the relative ratios remain large because DRAM physics is dominated by off-chip wire capacitance and cell charge dynamics, not transistor size):

```
Operation                    Energy (approx, 45nm)   At 3nm (est.)
─────────────────────────    ────────────────────    ─────────────
8-bit integer ADD            ~0.03 pJ                ~0.003 pJ
32-bit float MUL             ~3.7 pJ                 ~0.5–1 pJ
Read 32 bits from SRAM       ~5 pJ                   ~0.5–1 pJ
Read 32 bits from DRAM       ~640 pJ                 ~100–200 pJ

At 3nm, compute and SRAM costs drop ~5–10× (voltage scaling, shorter wires).
DRAM cost drops less (~3–6×) because off-chip IO capacitance dominates.
Net ratio at 3nm: DRAM still ~100–400× more expensive than arithmetic.
The qualitative lesson — DRAM is catastrophically expensive — remains true
at every process node. The 170× figure is illustrative; the principle is invariant.
```

This means: **the entire purpose of GPU/NPU architecture is to avoid going to DRAM.** Every cache, every coalescing unit, every tiling strategy, every compression engine exists because of this energy gap. The ALUs are almost free by comparison.

tiny-gpu makes this visible by having **no caches, no coalescing, no tiling** — so every operation pays the full DRAM cost. Real chips spend billions of transistors to avoid that cost.

### Visual: Energy Cost Comparison

```
Energy per operation (relative to INT8 ADD = 1×)

  INT8 ADD       │█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  1×   ~0.03 pJ
  FP32 MUL       │███████████████████████████░░░░░░░░░░░░░░░░░░░│  123×  ~3.7 pJ
  SRAM read      │████████████████████████████████████░░░░░░░░░░│  167×  ~5 pJ
  DRAM read      │████████████████████████████████████████████████████████████████████████████████████████████████░│  21,333×  ~640 pJ
                 0                                                                                              21,333×

  ┌─────────────────────────────────────────────────────────────────┐
  │  DRAM read  ≈  170× more expensive than FP32 multiply           │
  │  DRAM read  ≈  128× more expensive than SRAM read               │
  │  → Every cache, coalescer, and tiling strategy exists for THIS gap  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Chapter 2: The Roofline Model

The **Roofline model** tells you whether a workload is limited by compute or by memory bandwidth.

```
Performance (ops/sec)

        │         ╱ Peak Compute (flat ceiling)
        │        ╱
        │       ╱
        │      ╱
        │     ╱
        │    ╱  ← Bandwidth-limited region (slope = memory bandwidth)
        │   ╱
        │  ╱
        │ ╱
        │╱
        └──────────────────────────────────
          Operational Intensity (ops/byte)
```

**Operational intensity** = how many operations you perform per byte you fetch from memory.

- **Low intensity** (left side): You fetch lots of data but do little math per byte. You're **bandwidth-limited**. Adding more ALUs won't help.
- **High intensity** (right side): You do lots of math per byte fetched. You're **compute-limited**. Adding more bandwidth won't help.

**LLM decode (generating tokens one at a time) is almost always on the left side — bandwidth-limited.** In the batch=1 dense case, each token requires streaming each layer's weight matrix once — and with no cross-token reuse, the weights must be re-fetched from DRAM every token. Modern engines reduce this through layer-by-layer tiling, kernel fusion, and KV cache management, but the fundamental per-token weight-read requirement remains. The operational intensity is roughly:

```
For dense decode at batch=1 (no persistent weight tiling):
  Operations per token: ~2 × parameter_count (one multiply + one accumulate per weight)
  Bytes read per token: ~parameter_count × bytes_per_weight
    + KV cache reads   (grows with context length: 2 × layers × heads × head_dim × seq_len)
    + dequant scaling  (scale factors per quantization group)

  Weight-only operational intensity ≈ 2 ops / bytes_per_weight
  Total intensity (including KV + aux) is lower, especially at long context

  At INT4 (0.5 bytes per weight): weight intensity ≈ 4 ops/byte
  At INT8 (1 byte per weight):    weight intensity ≈ 2 ops/byte
  At FP16 (2 bytes per weight):   weight intensity ≈ 1 op/byte

  Note: At long context (e.g. 32K tokens), KV cache reads can rival
  weight reads in total bytes, pushing effective AI below these values.
```

### Concrete Roofline Example

```
  Performance
  (TOPS/s)         ╔══════════════════════════════════════╗  ← Peak Compute Ceiling
                   ║  COMPUTE-LIMITED region              ║
  ─ ─ ─ ─ ─ ─ ─ ─ ╚═══════╗                             ║
                           ║╲                            ║
                           ║  ╲                          ║
                           ║    ╲  Bandwidth Roof        ║
  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─║─ ─ ─╲─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─║
                           ║ BW-  ╲                      ║
     MEMORY-LIMITED        ║ slope ╲                     ║
     (LLM decode lives     ║        ╲                    ║
      here)                ║         ╲                   ║
  ───────────────────────────────────────────────────────────────────
  0    1    2    4    8   16   32   64  128  256  512     ops / byte
       ↑                                    ↑
       FP16 decode                          Well-tiled GEMM
       (~1 op/byte)                         (100+ ops/byte)
```

LLM Decode Operational Intensity by Precision:

```
  Precision   ops/byte  Regime
  ─────────────────────────────
  FP16        ~1        Bandwidth-bound
  INT8        ~2        Bandwidth-bound
  INT4/FP4    ~4        Bandwidth-bound
  Tiled GEMM  100+      Compute-bound
```

These are all **very low** operational intensities for decode. For comparison, a well-tiled matrix-matrix multiply can achieve 100+ ops/byte. LLM decode is firmly in the bandwidth-limited regime on every platform.

### The Hard Upper Bound

For LLM decode, the hard upper bound on performance is:

```
tokens/sec ≤ DRAM_bandwidth / bytes_of_weights_read_per_token
```

This is the single most important equation in inference hardware. We will apply it throughout this document.

---

## Chapter 3: Why GPUs Exist

A **CPU** is designed to do **one thing at a time, very fast, with complex logic** — branch prediction, out-of-order execution, speculative execution, deep caches.

A **GPU** is designed to do **thousands of simple things at the same time** — many small ALUs, massive parallelism, simple control logic.

Why? Because graphics (and now AI) are **embarrassingly parallel** problems:

```
Rendering a 1920×1080 image = 2,073,600 pixels
Each pixel needs roughly the same calculation
A CPU would do them one at a time: 2 million iterations
A GPU does thousands at once: ~2000 iterations with 1000 parallel threads
```

Imagine you need to add two arrays of 8 numbers:

```
A = [0, 1, 2, 3, 4, 5, 6, 7]
B = [0, 1, 2, 3, 4, 5, 6, 7]
C = A + B = [0, 2, 4, 6, 8, 10, 12, 14]
```

**CPU approach:** One core, one loop, 8 iterations. Sequential.

```python
for i in range(8):
    C[i] = A[i] + B[i]  # iteration 0, then 1, then 2...
```

**GPU approach:** 8 threads, all at once. Each thread computes ONE element.

```
Thread 0: C[0] = A[0] + B[0]  ─┐
Thread 1: C[1] = A[1] + B[1]   │
Thread 2: C[2] = A[2] + B[2]   │  ALL AT THE
Thread 3: C[3] = A[3] + B[3]   ├  SAME TIME
Thread 4: C[4] = A[4] + B[4]   │
Thread 5: C[5] = A[5] + B[5]   │
Thread 6: C[6] = A[6] + B[6]   │
Thread 7: C[7] = A[7] + B[7]  ─┘
```

The key insight: **trade single-thread speed for massive parallelism.**

```
CPU:  4 cores  × very fast  × very complex = great for sequential work
GPU:  1000s of cores × slower × simpler    = great for parallel work
```

---

## Chapter 4: What Is Verilog / SystemVerilog?

Verilog is **not** a programming language like Python or C. It's a **hardware description language (HDL)** — it describes physical circuits.

When you write this in Python:
```python
c = a + b  # This runs on an existing CPU
```

When you write this in Verilog:
```sv
assign c = a + b;  // This CREATES a physical adder circuit
```

The Python runs as software on hardware that already exists. The Verilog **becomes** hardware.

Here's the tiny-gpu ALU in SystemVerilog (`alu.sv`):

```sv
case (decoded_alu_arithmetic_mux)
    ADD: alu_out_reg <= rs + rt;
    SUB: alu_out_reg <= rs - rt;
    MUL: alu_out_reg <= rs * rt;
    DIV: alu_out_reg <= rs / rt;
endcase
```

This doesn't run on a computer. This **describes a circuit** that, when manufactured, will physically add/subtract/multiply/divide numbers using transistors and wires.

---

## Chapter 5: From Verilog to Real Chips

```
1. Write SystemVerilog     ← tiny-gpu is here (src/*.sv)
       ↓
2. Synthesize              ← Convert HDL to logic gates
       ↓                     (tools: Yosys, Synopsys Design Compiler)
3. Place & Route           ← Arrange gates on silicon, draw wires
       ↓                     (tools: OpenLane, Cadence Innovus)
4. Generate GDS            ← Final layout file ← tiny-gpu has these! (gds/*.gds)
       ↓
5. Fabrication             ← Send GDS to a foundry (TSMC, Samsung, GlobalFoundries)
       ↓                     They etch the pattern into silicon wafers
6. Physical Chip           ← Package, test, ship
```

tiny-gpu has **GDS files** in the repo (`gds/0/gpu.gds`, `gds/1/gpu.gds`). GDS (Graphic Data System) is the final physical layout that a foundry uses to manufacture the chip. These were likely generated through the **Tiny Tapeout** program using the open-source **Sky130 PDK** from SkyWater Technology at 130nm.

For comparison:
- tiny-gpu GDS: **130nm** process, probably a few thousand transistors
- Snapdragon 8 Elite (Adreno 830): **3nm** process, ~20 billion transistors
- NVIDIA B200: **4nm** process, ~208 billion transistors

### What the ALU Becomes in Silicon

When `alu.sv` goes through synthesis, it becomes physical circuits:

```
1. rs + rt  → An 8-bit ripple-carry adder (or carry-lookahead adder)
              = ~40-80 transistors

2. rs - rt  → An 8-bit subtractor (adder with inverted input + carry-in)
              = ~40-80 transistors

3. rs * rt  → An 8-bit multiplier (array multiplier or Booth multiplier)
              = ~200-500 transistors

4. rs / rt  → An 8-bit divider (restoring or non-restoring division)
              = ~500-1000 transistors

5. The case/mux → A 4-to-1 multiplexer selecting which result to output
              = ~30 transistors

6. alu_out_reg <= → A register (8 flip-flops) that stores the result
              = ~48 transistors (6 transistors per flip-flop × 8 bits)
```

Total for one ALU: **~800-1700 transistors.** tiny-gpu has 8 ALUs (2 cores × 4 threads), so ~6,000-14,000 transistors just for ALUs. The entire tiny-gpu is probably **~50,000-100,000 transistors.**

---

# Part II: tiny-gpu Architecture

## Chapter 6: The Programming Model — SIMD

**Same Instruction, Multiple Data.** Every thread runs the *exact same program*, but each thread knows its own identity:

- `%blockIdx` — which block am I in?
- `%blockDim` — how many threads per block?
- `%threadIdx` — which thread am I within my block?

Here's the matrix addition kernel:

```asm
MUL R0, %blockIdx, %blockDim
ADD R0, R0, %threadIdx         ; i = blockIdx * blockDim + threadIdx

CONST R1, #0                   ; baseA
CONST R2, #8                   ; baseB
CONST R3, #16                  ; baseC

ADD R4, R1, R0                 ; addr(A[i])
LDR R4, R4                     ; load A[i]

ADD R5, R2, R0                 ; addr(B[i])
LDR R5, R5                     ; load B[i]

ADD R6, R4, R5                 ; C[i] = A[i] + B[i]

ADD R7, R3, R0                 ; addr(C[i])
STR R7, R6                     ; store C[i]

RET
```

Every thread runs this exact program. The only thing that differs is the three special read-only registers. For 8 threads across 2 blocks of 4:

| Thread | blockIdx | blockDim | threadIdx | `i = blockIdx * blockDim + threadIdx` |
|--------|----------|----------|-----------|---------------------------------------|
| 0 | 0 | 4 | 0 | **0** |
| 1 | 0 | 4 | 1 | **1** |
| 2 | 0 | 4 | 2 | **2** |
| 3 | 0 | 4 | 3 | **3** |
| 4 | 1 | 4 | 0 | **4** |
| 5 | 1 | 4 | 1 | **5** |
| 6 | 1 | 4 | 2 | **6** |
| 7 | 1 | 4 | 3 | **7** |

Same code → different `i` → different data → **SIMD**.

---

## Chapter 7: Top-Level Architecture — gpu.sv

From `gpu.sv`:

```sv
parameter NUM_CORES = 2,
parameter THREADS_PER_BLOCK = 4,
parameter DATA_MEM_NUM_CHANNELS = 4,
parameter PROGRAM_MEM_NUM_CHANNELS = 1
```

```
┌─────────────────────────────────────────────────────────────────┐
│                           gpu.sv                                │
│                                                                 │
│  ┌─────────┐  ┌────────────┐  ┌───────────────────────────┐    │
│  │  DCR    │  │ Dispatcher  │  │  Program Memory Controller│    │
│  │         │  │             │  │  (1 channel)              │    │
│  │ stores  │  │ assigns     │  └───────────────────────────┘    │
│  │ thread  │  │ blocks to  │                                    │
│  │ count   │  │ free cores │  ┌───────────────────────────┐    │
│  └─────────┘  └────────────┘  │  Data Memory Controller   │    │
│                                │  (4 channels)             │    │
│  ┌──────────────────────────┐ └───────────────────────────┘    │
│  │         Core 0           │                                   │
│  │  ┌fetcher┐ ┌decoder┐    │                                   │
│  │  │ Thread 0 │ Thread 1 │ │                                   │
│  │  │ Thread 2 │ Thread 3 │ │                                   │
│  └──────────────────────────┘                                   │
│  ┌──────────────────────────┐                                   │
│  │         Core 1           │                                   │
│  │  ┌fetcher┐ ┌decoder┐    │                                   │
│  │  │ Thread 0 │ Thread 1 │ │                                   │
│  │  │ Thread 2 │ Thread 3 │ │                                   │
│  └──────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

Each thread has its own:
- **ALU** (`alu.sv`) — does math
- **LSU** (`lsu.sv`) — loads/stores from memory
- **Register file** (`registers.sv`) — holds data (16 registers, 8 bits each)
- **PC** (`pc.sv`) — tracks which instruction to execute next

But all threads in a core share:
- **One fetcher** (`fetcher.sv`) — fetches the current instruction
- **One decoder** (`decoder.sv`) — decodes it into control signals
- **One scheduler** (`scheduler.sv`) — manages the execution state machine

This means **all threads execute the same instruction at the same time.** That's SIMD.

---

## Chapter 8: The Device Control Register — dcr.sv

The simplest module. It stores one number: how many threads to launch.

```sv
reg [7:0] device_conrol_register;
assign thread_count = device_conrol_register[7:0];

always @(posedge clk) begin
    if (reset) begin
        device_conrol_register <= 8'b0;
    end else begin
        if (device_control_write_enable) begin
            device_conrol_register <= device_control_data;
        end
    end
end
```

One register. One write port. Stores `thread_count = 8`.

---

## Chapter 9: The Dispatcher — dispatch.sv

The dispatcher divides work into blocks and assigns them to available cores:

```sv
wire [7:0] total_blocks;
assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
// (8 + 4 - 1) / 4 = 2 blocks
```

Initial dispatch:

```sv
if (blocks_dispatched < total_blocks) begin
    core_start[i] <= 1;
    core_block_id[i] <= blocks_dispatched;
    core_thread_count[i] <= (blocks_dispatched == total_blocks - 1)
        ? thread_count - (blocks_dispatched * THREADS_PER_BLOCK)
        : THREADS_PER_BLOCK;
    blocks_dispatched = blocks_dispatched + 1;
end
```

Completion detection:

```sv
if (core_start[i] && core_done[i]) begin
    core_reset[i] <= 1;
    core_start[i] <= 0;
    blocks_done = blocks_done + 1;
end

if (blocks_done == total_blocks) begin
    done <= 1;
end
```

This is **dynamic work scheduling** — if you had 6 blocks and 2 cores, the first 2 blocks go out immediately, and as each core finishes, it gets the next block.

---

## Chapter 10: The Compute Core — core.sv

Each core processes one block at a time. From `core.sv`, the thread units are generated with a `generate` loop:

```sv
genvar i;
generate
    for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads
        alu alu_instance (
            .enable(i < thread_count),
            .rs(rs[i]),
            .rt(rt[i]),
            .alu_out(alu_out[i])
            // ... but ALL share the same decoded_alu_arithmetic_mux
        );

        lsu lsu_instance (...);

        registers #(
            .THREAD_ID(i),  // THIS makes each thread unique
        ) register_instance (...);

        pc pc_instance (...);
    end
endgenerate
```

The critical insight: **all threads share the same decoder output** (same instruction, same control signals), but each thread has **its own data path** (own registers, own ALU, own LSU).

When the decoder says "ADD R0, R0, %threadIdx":
- Thread 0's ALU computes `0 + 0 = 0`
- Thread 1's ALU computes `0 + 1 = 1`
- Thread 2's ALU computes `0 + 2 = 2`
- Thread 3's ALU computes `0 + 3 = 3`

All in the **same clock cycle**, driven by the **same control signals**.

---

## Chapter 11: The Scheduler — scheduler.sv

The scheduler is a 7-state finite state machine:

```
IDLE → FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE → (FETCH or DONE)
```

```sv
case (core_state)
    IDLE: begin
        if (start) core_state <= FETCH;
    end
    FETCH: begin
        if (fetcher_state == 3'b010) core_state <= DECODE;
    end
    DECODE: begin
        core_state <= REQUEST;
    end
    REQUEST: begin
        core_state <= WAIT;
    end
    WAIT: begin
        // THE CRITICAL BOTTLENECK:
        reg any_lsu_waiting = 1'b0;
        for (int i = 0; i < THREADS_PER_BLOCK; i++) begin
            if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10) begin
                any_lsu_waiting = 1'b1;
                break;
            end
        end
        if (!any_lsu_waiting) core_state <= EXECUTE;
        // OTHERWISE: ENTIRE CORE STALLS. ALL ALUs IDLE.
    end
    EXECUTE: begin
        core_state <= UPDATE;
    end
    UPDATE: begin
        if (decoded_ret) begin
            done <= 1;
            core_state <= DONE;
        end else begin
            // TODO: Branch divergence. Assume all next_pc converge
            current_pc <= next_pc[THREADS_PER_BLOCK-1];
            core_state <= FETCH;
        end
    end
endcase
```

**This is the most important module in the entire project for understanding GPU architecture.** The WAIT state makes the memory bottleneck physically visible.

---

## Chapter 12: The Instruction Fetcher — fetcher.sv

The fetcher has a 3-state FSM:

```sv
IDLE: begin
    if (core_state == 3'b001) begin        // Core says FETCH
        fetcher_state <= FETCHING;
        mem_read_valid <= 1;
        mem_read_address <= current_pc;
    end
end
FETCHING: begin
    if (mem_read_ready) begin              // Memory controller responded
        fetcher_state <= FETCHED;
        instruction <= mem_read_data;
        mem_read_valid <= 0;
    end
end
FETCHED: begin
    if (core_state == 3'b010) begin        // Core moved to DECODE
        fetcher_state <= IDLE;
    end
end
```

The fetcher doesn't talk directly to program memory — it talks to the **program memory controller**, which arbitrates between both cores' fetchers (since there's only 1 channel).

---

## Chapter 13: The Decoder — decoder.sv

The 16-bit instruction format:

```
[15:12]  [11:8]  [7:4]  [3:0]
opcode    rd      rs     rt
```

The decoder is a `case` statement that fires on the DECODE state:

```sv
case (instruction[15:12])
    ADD: begin
        decoded_reg_write_enable <= 1;
        decoded_reg_input_mux <= 2'b00;       // Source = ALU output
        decoded_alu_arithmetic_mux <= 2'b00;  // ALU operation = ADD
    end
    LDR: begin
        decoded_reg_write_enable <= 1;
        decoded_reg_input_mux <= 2'b01;       // Source = MEMORY output
        decoded_mem_read_enable <= 1;
    end
    STR: begin
        decoded_mem_write_enable <= 1;
    end
    CONST: begin
        decoded_reg_write_enable <= 1;
        decoded_reg_input_mux <= 2'b10;       // Source = IMMEDIATE value
    end
    CMP: begin
        decoded_alu_output_mux <= 1;           // ALU outputs NZP flags
        decoded_nzp_write_enable <= 1;
    end
    BRnzp: begin
        decoded_pc_mux <= 1;                   // PC uses branch logic
    end
    RET: begin
        decoded_ret <= 1;
    end
endcase
```

The key multiplexer `decoded_reg_input_mux` selects WHERE the register gets its new value:

```
00 → ALU output    (ADD, SUB, MUL, DIV results)
01 → Memory output (LDR results)
10 → Immediate     (CONST value)
```

---

## Chapter 14: The Register File — registers.sv

Each thread has 16 registers, 8 bits each. The magic is in the parameters:

```sv
module registers #(
    parameter THREADS_PER_BLOCK = 4,
    parameter THREAD_ID = 0,        // ← Different for each thread instance!
)
```

On reset:

```sv
registers[13] <= 8'b0;              // %blockIdx (updated dynamically)
registers[14] <= THREADS_PER_BLOCK; // %blockDim = 4 (constant)
registers[15] <= THREAD_ID;         // %threadIdx = 0, 1, 2, or 3
```

`THREAD_ID` is a **synthesis-time parameter**. When the `generate` loop creates 4 register file instances, it passes `THREAD_ID(0)`, `THREAD_ID(1)`, `THREAD_ID(2)`, `THREAD_ID(3)`. These become **different physical circuits** with different hardwired values.

The write-back logic:

```sv
if (core_state == 3'b110) begin  // UPDATE
    if (decoded_reg_write_enable && decoded_rd_address < 13) begin
        case (decoded_reg_input_mux)
            ARITHMETIC: registers[decoded_rd_address] <= alu_out;
            MEMORY:     registers[decoded_rd_address] <= lsu_out;
            CONSTANT:   registers[decoded_rd_address] <= decoded_immediate;
        endcase
    end
end
```

Note `decoded_rd_address < 13` — you can't write to the read-only registers.

---

## Chapter 15: The ALU — alu.sv

Each thread has a dedicated ALU. Two modes:

**Arithmetic** (when `decoded_alu_output_mux == 0`):
```sv
case (decoded_alu_arithmetic_mux)
    ADD: alu_out_reg <= rs + rt;
    SUB: alu_out_reg <= rs - rt;
    MUL: alu_out_reg <= rs * rt;
    DIV: alu_out_reg <= rs / rt;
endcase
```

**Comparison** (when `decoded_alu_output_mux == 1`, for `CMP`):
```sv
alu_out_reg <= {5'b0, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)};
//                     ↑ Positive      ↑ Zero          ↑ Negative
```

The ALU is the cheapest part of the whole system. **The ALU is not the bottleneck.** It computes in 1 cycle but the core spends ~7 cycles per instruction on everything else.

---

## Chapter 16: The Program Counter — pc.sv

By default, `next_pc = current_pc + 1`. But on a `BRnzp` instruction:

```sv
if (decoded_pc_mux == 1) begin
    if ((nzp & decoded_nzp) != 3'b0) begin
        next_pc <= decoded_immediate;  // Branch taken
    end else begin
        next_pc <= current_pc + 1;     // Branch not taken
    end
end
```

**Important limitation:** tiny-gpu assumes all threads branch to the same PC (`current_pc <= next_pc[THREADS_PER_BLOCK-1]`). No branch divergence support.

---

## Chapter 17: The Load-Store Unit — lsu.sv

The LSU has a 4-state FSM:

```
IDLE → REQUESTING → WAITING → DONE
```

For a `LDR` (load):
```sv
REQUESTING: begin
    mem_read_valid <= 1;
    mem_read_address <= rs;
    lsu_state <= WAITING;
end
WAITING: begin
    if (mem_read_ready == 1) begin
        mem_read_valid <= 0;
        lsu_out <= mem_read_data;
        lsu_state <= DONE;
    end
end
```

One request at a time. No pipelining. No prefetching. The LSU sends a request, waits for a response, and only then can the core proceed.

---

## Chapter 18: The Memory Controller — controller.sv

The controller arbitrates between multiple consumers competing for limited memory channels:

```sv
IDLE: begin
    for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin
        if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin
            channel_serving_consumer[j] = 1;
            current_consumer[i] <= j;
            mem_read_valid[i] <= 1;
            mem_read_address[i] <= consumer_read_address[j];
            controller_state[i] <= READ_WAITING;
            break;  // First-come, first-served by index
        end
    end
end
```

This is a **priority arbiter with fixed ordering**. Consumer 0 always gets checked first. Consumer 7 gets checked last.

The `channel_serving_consumer` bitmask prevents two channels from picking up the same request:

```sv
reg [NUM_CONSUMERS-1:0] channel_serving_consumer;
```

Note this uses **blocking assignment** (`=` not `<=`), so within the same clock cycle, if Channel 0 sets `channel_serving_consumer[0] = 1`, Channel 1 will see it immediately and skip Consumer 0.

---

## Chapter 19: The ISA

11 instructions, all 16 bits:

| Opcode | Instruction | What it does |
|--------|------------|--------------|
| `0000` | `NOP` | Nothing |
| `0001` | `BRnzp` | Branch if NZP condition matches |
| `0010` | `CMP` | Compare rs and rt, set NZP flags |
| `0011` | `ADD` | rd = rs + rt |
| `0100` | `SUB` | rd = rs - rt |
| `0101` | `MUL` | rd = rs × rt |
| `0110` | `DIV` | rd = rs ÷ rt |
| `0111` | `LDR` | rd = memory[rs] |
| `1000` | `STR` | memory[rs] = rt |
| `1001` | `CONST` | rd = immediate (bits [7:0]) |
| `1111` | `RET` | Thread done |

16 registers per thread: `R0`-`R12` (read/write), `R13` = `%blockIdx`, `R14` = `%blockDim`, `R15` = `%threadIdx` (read-only).

---

# Part II.5: Complete Code Walkthrough — Line-by-Line with Hardware Mapping

This section provides **line-by-line explanations of every critical tiny-gpu code section** with concrete examples, execution timelines, and detailed mappings to Snapdragon 8 Elite and NVIDIA B200 architectures. This is where abstract GPU concepts become concrete SystemVerilog reality.

## Chapter 19.5: gpu.sv — Two Independent Memory Controllers

### Why Separate Program and Data Controllers?

```sv
// Program memory controller (read-only)
controller #(
    .NUM_CHANNELS(PROGRAM_MEM_NUM_CHANNELS),
    .WRITE_ENABLE(0)  // Read-only!
) program_memory_controller (...)

// Data memory controller (read + write)
controller #(
    .NUM_CHANNELS(DATA_MEM_NUM_CHANNELS),
    .WRITE_ENABLE(1)  // Read and write
) data_memory_controller (...)
```

**Design rationale:**
- Program memory: Instructions loop in kernels → cacheable → 1 channel sufficient
- Data memory: Unpredictable workload → needs 4 channels for parallelism
- Separation enables independent optimization

---

## Chapter 19.6: dispatch.sv — Block Distribution Algorithm

### Ceiling Division for Block Count

```sv
wire [7:0] total_blocks;
assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
```

**Examples:**
```
8 threads, 4/block: (8+3)/4 = 11/4 = 2 blocks
7 threads, 4/block: (7+3)/4 = 10/4 = 2 blocks (last block has 3 threads)
```

### The Core Dispatch Loop

```sv
if (blocks_dispatched < total_blocks) begin
    core_block_id[i] <= blocks_dispatched;
    core_thread_count[i] <= (blocks_dispatched == total_blocks - 1)
        ? thread_count - (blocks_dispatched * THREADS_PER_BLOCK)
        : THREADS_PER_BLOCK;
    blocks_dispatched = blocks_dispatched + 1;
end
```

**Key insight:** Last block might have fewer threads (handle with ternary operator)

---

## Chapter 19.7: scheduler.sv — The 7-State FSM and WAIT Bottleneck

### State Machine Definition

```sv
localparam IDLE = 3'b000, FETCH = 3'b001, DECODE = 3'b010, REQUEST = 3'b011,
           WAIT = 3'b100, EXECUTE = 3'b101, UPDATE = 3'b110, DONE = 3'b111;
```

### The WAIT State: Where tiny-gpu Stalls

```sv
WAIT: begin
    reg any_lsu_waiting = 1'b0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) begin
        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10) begin
            any_lsu_waiting = 1'b1;
            break;
        end
    end
    if (!any_lsu_waiting) core_state <= EXECUTE;
    // OTHERWISE: CORE STALLS, ALL ALUs IDLE
end
```

**Timeline for LDR (100-cycle DRAM latency):**
```
Cycle 0-5:   Simple instructions (compute phase)
Cycle 6-105: WAIT state (memory latency)
Cycle 106+:  Resume execution
Total: 100 wasted cycles per load!
```

### Comparison to Real Hardware

```
tiny-gpu: 1 wave stalled → entire core stalls → 0.8% ALU utilization
Snapdragon: 40 waves stalled → switch to other waves → 60-80% utilization
B200: 3 warps stalled → switch to other warps → 85%+ utilization
```

---

## Chapter 19.8: fetcher.sv — Async Instruction Fetch Pattern

### 3-State FSM

```sv
case (fetcher_state)
    IDLE: begin
        if (core_state == 3'b001) begin  // FETCH state
            fetcher_state <= FETCHING;
            mem_read_valid <= 1;
            mem_read_address <= current_pc;
        end
    end

    FETCHING: begin
        if (mem_read_ready) begin
            fetcher_state <= FETCHED;
            instruction <= mem_read_data;
            mem_read_valid <= 0;
        end
    end

    FETCHED: begin
        if (core_state == 3'b010) begin  // DECODE state
            fetcher_state <= IDLE;
        end
    end
endcase
```

**Request/response handshake:**
```
Cycle N:   Request: mem_read_valid=1, address=PC
Cycles N+1 to N+100: Wait for DRAM
Cycle N+101: Response: mem_read_ready=1, instruction latched
Cycle N+102: Stop requesting, move to DECODE
```

---

## Chapter 19.9: decoder.sv — Instruction to Control Signals

### 11-Instruction Decoding

```sv
case (instruction[15:12])
    4'b0011: begin  // ADD
        decoded_reg_write_enable <= 1;
        decoded_reg_input_mux <= 2'b00;
        decoded_alu_arithmetic_mux <= 2'b00;
    end
    4'b0111: begin  // LDR
        decoded_reg_write_enable <= 1;
        decoded_reg_input_mux <= 2'b01;    // LSU output
        decoded_mem_read_enable <= 1;
    end
    // ... 9 more instructions ...
endcase
```

**All threads get same control signals (SIMD)**

---

## Chapter 19.10: registers.sv — SIMD via Special Registers

### The Three Read-Only Registers

```sv
registers[13] <= block_id;          // %blockIdx
registers[14] <= THREADS_PER_BLOCK; // %blockDim = 4
registers[15] <= THREAD_ID;         // %threadIdx = 0,1,2,3
```

### Why SIMD Works

```
All threads execute: ADD R0, R0, %threadIdx

Thread 0: R0 = old_R0 + 0 = old_R0
Thread 1: R0 = old_R0 + 1 = old_R0 + 1
Thread 2: R0 = old_R0 + 2 = old_R0 + 2
Thread 3: R0 = old_R0 + 3 = old_R0 + 3

Same instruction, different thread-local register → different result!
```

### Register File Comparison

```
tiny-gpu:     64 bytes per core (exactly known; it's in the RTL)
Snapdragon:   ~7.8 MB per GPU (rough estimate: ~650 KB per CU × 12 CUs)
              [⚠ SPECULATIVE: Qualcomm has not publicly confirmed register file sizes.
               This is derived from wave count × register file per wave assumptions.]
Ratio:        ~120,000× larger (approximate)

Why? Snapdragon keeps 40 waves resident simultaneously.
```

---

## Chapter 19.11: alu.sv — Compute is "Free"

### 4 Basic Operations

```sv
case (decoded_alu_arithmetic_mux)
    2'b00: alu_out_reg <= rs + rt;  // ADD
    2'b01: alu_out_reg <= rs - rt;  // SUB
    2'b10: alu_out_reg <= rs * rt;  // MUL
    2'b11: alu_out_reg <= rs / rt;  // DIV
endcase
```

### CMP (Compare) for Branching

```sv
if (decoded_alu_output_mux == 1) begin  // CMP
    alu_out_reg <= {5'b0, rs > rt, rs == rt, rs < rt};
end

// Later: BRnzp checks NZP flags to determine branch
```

### Energy Profile

```
Operation        Energy          Relative
────────────────────────────────────────
32-bit ADD       ~0.03 pJ        1×
32-bit MUL       ~3.7 pJ         123×
DRAM read        ~640 pJ         21,333×!

Implication: Hardware designers obsess over memory, not compute.
```

---

## Chapter 19.12: lsu.sv — Async Memory: The Stall Point

### 4-State Memory FSM

```sv
case (lsu_state)
    IDLE: begin
        if (decoded_mem_read_enable && core_state == 3'b011) begin
            lsu_state <= REQUESTING;
        end
    end

    REQUESTING: begin
        mem_read_valid <= 1;
        mem_read_address <= rs;
        lsu_state <= WAITING;
    end

    WAITING: begin
        if (mem_read_ready == 1) begin
            mem_read_valid <= 0;
            lsu_out <= mem_read_data;
            lsu_state <= DONE;
        end
        // Otherwise: STALL (stay in WAITING)
    end

    DONE: begin
        if (core_state == 3'b110) begin
            lsu_state <= IDLE;
        end
    end
endcase
```

### LDR Timeline (100 cycles DRAM latency)

```
Cycle N:     LSU IDLE → REQUESTING
Cycle N+1:   mem_read_valid ← 1, address ← rs
             LSU REQUESTING → WAITING
             scheduler detects waiting → core_state = WAIT

Cycles N+2 to N+101: (100 cycle DRAM latency)
             LSU stays WAITING
             core_state stays WAIT
             ALL ALUs IDLE!

Cycle N+102: mem_read_ready ← 1
             LSU WAITING → DONE
             lsu_out ← data

Cycle N+103: core_state ← EXECUTE
Cycle N+104: core_state ← UPDATE, register written
             LSU DONE → IDLE

Total: 105 cycles per load
```

---

## Chapter 19.13: controller.sv — Fixed-Priority Arbiter (Unfair!)

### The Priority Scan (Loop Over Consumers)

```sv
for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin
    if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin
        channel_serving_consumer[j] = 1;
        current_consumer[i] <= j;
        mem_read_valid[i] <= 1;
        mem_read_address[i] <= consumer_read_address[j];
        controller_state[i] <= READ_WAITING;
        break;  // STOP AFTER FINDING FIRST REQUEST!
    end
end
```

**Problem: Fixed Priority = Starvation**

```
If Consumer 0 constantly requests:
  Cycle 0-100: Channel picks Consumer 0
  Cycle 101-200: Channel picks Consumer 0 again
  Cycle 201+: Consumer 0 still requesting → picked again!

Consumer 1's request (issued Cycle 10):
  Actually served at: Cycle 300+
  Latency: 290 cycles instead of 100!
```

### 5-State Controller FSM (Per Channel)

```
IDLE → READ_WAITING → READ_RELAYING → IDLE
     → WRITE_WAITING → WRITE_RELAYING → IDLE

Each state waits for handshake before transitioning.
```

---

## Chapter 19.14: Memory Coalescing Opportunities

### Best Case: Stride-1 (Sequential Access)

```
4 threads loading A[0], A[1], A[2], A[3]:

Without coalescing:
  Transaction 0: Read address 0 (1 byte)
  Transaction 1: Read address 1 (1 byte)
  Transaction 2: Read address 2 (1 byte)
  Transaction 3: Read address 3 (1 byte)
  Total: 4 DRAM accesses

With coalescing:
  Addresses [0-3] fit in single 64-byte cache line
  Single transaction: Read cache line
  Distribute to threads
  Total: 1 DRAM access

Benefit: 4× fewer accesses, 4× less power, 4× less latency!
```

### Worst Case: Stride-64

```
4 threads loading A[0], A[64], A[128], A[192]:

Stride = 64 bytes (cache line size)
Each address hits different cache line
Coalescing cannot help
Still: 4 DRAM accesses

This is why GPU kernels must optimize memory access patterns!
```

---

## Chapter 19.15: Full Execution Trace — Matrix Add Kernel

### The Kernel

```
A[0-7] = [0,1,2,3,4,5,6,7]
B[0-7] = [0,1,2,3,4,5,6,7]
C[0-7] = A[i] + B[i] = [0,2,4,6,8,10,12,14]

8 threads = 2 blocks × 4 threads/block
```

### Cycle Timeline for Thread 0

```
Cycles 0-7:    MUL R0, blockIdx, blockDim
Cycles 8-14:   ADD R0, R0, threadIdx
Cycles 15-21:  CONST R1, #0
... (similar for other CONSTs and ADDs) ...

Cycle 63:      LDR R4, R4 (FIRST LOAD)
               LSU: IDLE → REQUESTING

Cycle 64:      mem_read_valid ← 1
               LSU: REQUESTING → WAITING
               **CORE STALLED!**

Cycles 65-164: (100 cycles DRAM latency)
               mem_read_ready = 0
               ALUs IDLE

Cycle 165:     mem_read_ready ← 1
               Data arrives
               LSU: WAITING → DONE

Cycle 166-171: Housekeeping

Cycle 172:     LDR R5, R5 (SECOND LOAD)
               **ANOTHER 100-CYCLE STALL!**

Cycles 173-272: Second DRAM wait

Cycle 273+:    Final instructions (ADD, STR, RET)

Total: ~285 cycles per thread
```

### Speedup with Wave Scheduling (Snapdragon)

```
tiny-gpu: 1 thread stalled → entire core stalls
         ALU utilization: 30 useful cycles / 285 total = 10%

Snapdragon: 40 threads stalled → switch to other threads
            Thread 0 stalled → execute Thread 1-39
            Continuous execution until all stalled
            ALU utilization: 85%+

Speedup: 8.5× just from latency hiding!
         × 40× from 40 threads = 340× theoretical maximum
         Practical: 30-40× due to other factors
```

---

## Chapter 19.16: Energy Breakdown

### Per-Operation Energy Costs

```
Operation                   Energy        Per Operation
──────────────────────────────────────────────────────
8-bit integer ADD           0.03 pJ       Negligible
32-bit FP multiply          3.7 pJ        ~100 ADDs
32-bit SRAM read            5 pJ          ~170 ADDs
32-bit DRAM read            640 pJ        ~21,000 ADDs!
```

### Energy Per Matrix Add Kernel

```
Without optimizations:
  Instruction fetches:      ~200 pJ
  Register accesses:        ~30 pJ
  ALU operations:           ~50 pJ
  Controller overhead:      ~50 pJ
  Two DRAM reads:           ~1,280 pJ
  TOTAL:                    ~1,610 pJ

Optimization 1: Instruction cache (-150 pJ)
  Caches loop instructions locally

Optimization 2: Coalescing (-320 pJ)
  4 threads → 1 merged DRAM access

Optimization 3: Data cache (-400 pJ)
  Some loads hit L1 instead of DRAM

Optimization 4: Better arbitration (-50 pJ)
  No starvation delays

Optimized total:            ~690 pJ (57% reduction!)
```

---

# Part III: Execution Trace — Step by Step

## Chapter 20: Launching a Kernel

From `test/helpers/setup.py`:

```python
# 1. Load kernel code into program memory
program_memory.load(program)

# 2. Load data into data memory
data_memory.load(data)

# 3. Tell GPU how many threads
dut.device_control_write_enable.value = 1
dut.device_control_data.value = 8

# 4. GO
dut.start.value = 1
```

Memory layout after setup:

```
PROGRAM MEMORY (16-bit):
  Addr 0:  MUL R0, %blockIdx, %blockDim
  Addr 1:  ADD R0, R0, %threadIdx
  ...
  Addr 12: RET

DATA MEMORY (8-bit):
  Addr 0-7:   [0, 1, 2, 3, 4, 5, 6, 7]   ← Matrix A
  Addr 8-15:  [0, 1, 2, 3, 4, 5, 6, 7]   ← Matrix B
  Addr 16-23: [?, ?, ?, ?, ?, ?, ?, ?]    ← Matrix C (output)
```

---

## Chapter 21: Cycle-by-Cycle Execution of matadd

### Instruction 0: `MUL R0, %blockIdx, %blockDim`

**FETCH (Cycles 1-4):**

The fetcher requests instruction at PC=0 from program memory. The program memory controller has 1 channel and 2 consumers (Core 0's fetcher and Core 1's fetcher). Core 0 gets served first. Core 1 waits.

**DECODE (Cycle 5):**

```sv
// instruction = 0101_0000_1101_1110
//               MUL  R0   R13  R14
decoded_reg_write_enable <= 1;
decoded_reg_input_mux <= 2'b00;      // ALU
decoded_alu_arithmetic_mux <= 2'b10; // MUL
```

**REQUEST (Cycle 6):**

All 4 register files read `rs` and `rt`:
```
Thread 0: rs = registers[13] = 0 (blockIdx), rt = registers[14] = 4 (blockDim)
Thread 1: rs = 0, rt = 4
Thread 2: rs = 0, rt = 4
Thread 3: rs = 0, rt = 4
```

**WAIT (Cycle 7):** No memory access, passes through.

**EXECUTE (Cycle 8):** All 4 ALUs: `0 * 4 = 0` for all Core 0 threads.

**UPDATE (Cycle 9):** R0 = 0 for all Core 0 threads. PC → 1.

### Instruction 1: `ADD R0, R0, %threadIdx`

Same flow. After EXECUTE:

```
Core 0: Thread 0: 0+0=0, Thread 1: 0+1=1, Thread 2: 0+2=2, Thread 3: 0+3=3
Core 1: Thread 0: 4+0=4, Thread 1: 4+1=5, Thread 2: 4+2=6, Thread 3: 4+3=7
```

Now every thread has a **unique global ID** in R0.

### Instruction 6: `LDR R4, R4` — THE FIRST MEMORY LOAD

This is where everything gets interesting. All 4 LSUs in each core fire off memory requests simultaneously.

```
Core 0: Thread 0→addr 0, Thread 1→addr 1, Thread 2→addr 2, Thread 3→addr 3
Core 1: Thread 0→addr 4, Thread 1→addr 5, Thread 2→addr 6, Thread 3→addr 7
```

8 requests, 4 channels. The controller serves Core 0 first (fixed priority):

```
Cycle W+0: Ch0→C0T0, Ch1→C0T1, Ch2→C0T2, Ch3→C0T3. Core 1 BLOCKED.
Cycle W+1: Memory responds for Core 0
Cycle W+2: Controller relays to Core 0's LSUs
Cycle W+3: Channels free. Pick up Core 1's requests.
Cycle W+4: Memory responds for Core 1
Cycle W+5: Controller relays to Core 1's LSUs
```

**Core 1 starved for 3 extra cycles.**

### Final Result

After STR completes:

```
Address: 16  17  18  19  20  21  22  23
Data:     0   2   4   6   8  10  12  14
          ├── C = A + B ────────────────┤
```

---

# Part IV: The Five Problems

## Chapter 22: Instruction Fetch Bottleneck

**The problem:** From `fetcher.sv`, every instruction fetch goes through the 1-channel program memory controller. Two cores share this channel. Core 1 always waits for Core 0.

**Real solution:** Multi-level instruction caches. After the first execution of a loop body, subsequent iterations are served from on-chip instruction cache at much lower latency.

**Impact:** 13× fewer external memory fetches for the matadd kernel.

---

## Chapter 23: No Memory Coalescing

**The problem:** From `lsu.sv`, each thread sends its own independent memory request. 8 threads accessing addresses 0-7 = 8 separate transactions.

```
tiny-gpu — no coalescing:
────────────────────────────────────────────────────────────────
Thread 0 → addr 0  ─→  Memory request 0 ─→ [DRAM] → Thread 0
Thread 1 → addr 1  ─→  Memory request 1 ─→ [DRAM] → Thread 1
Thread 2 → addr 2  ─→  Memory request 2 ─→ [DRAM] → Thread 2
Thread 3 → addr 3  ─→  Memory request 3 ─→ [DRAM] → Thread 3
                       4 transactions, 4 round-trips
────────────────────────────────────────────────────────────────

Real GPU (Adreno 830 / B200) — coalesced:
────────────────────────────────────────────────────────────────
Thread 0 → addr 0 ─┐
Thread 1 → addr 1 ─┤─→ Coalescing unit ─→ 1 cache-line req ─→ [DRAM]
Thread 2 → addr 2 ─┤          │                                    │
Thread 3 → addr 3 ─┘          └──────────────────────────────────►│
                               1 transaction, 4× bandwidth saving  │
                               data distributed back to threads ◄──┘
────────────────────────────────────────────────────────────────
Qualcomm optimization guide: "sequential addresses from sequential
work-items = most efficient pattern" — because coalescing merges them.
```

**Real solution:** Coalescing units examine all threads' addresses within a wave and merge requests to adjacent addresses into fewer, wider transactions.

**Impact:** Up to 64× fewer memory transactions for stride-1 access patterns.

---

## Chapter 24: No Data Cache

**The problem:** Every load goes all the way to external memory. The second load (B[i]) costs the same as the first load (A[i]), even though the data is nearby.

**Real solution:** Multi-level cache hierarchy (L1 → L2 → SLC → DRAM). Second load hits L1 cache: ~4 cycles instead of ~100+ cycles.

**Impact:** 5-25× faster for repeated/nearby accesses.

---

## Chapter 25: Core Stalls on Memory

**The problem:** From `scheduler.sv`, the entire core stalls when ANY thread is waiting for memory. All ALUs idle.

**Real solution:** Wave/warp scheduling — when one group of threads stalls, the scheduler switches to another group. ALUs never idle.

**Impact:** 60-95× better ALU utilization for real workloads.

---

## Chapter 26: Unfair Memory Arbitration

**The problem:** From `controller.sv`, the fixed-priority scan always serves Core 0 first. Core 1 starves.

**Real solution:** QoS-aware arbitration with credit-based flow control. Each IP block gets fair bandwidth allocation.

**Impact:** No starvation. All compute units get fair memory bandwidth.

---

# Part V: Stride, Banking, and Conflicts

## Chapter 27: How the Controller Arbitrates

From `controller.sv`, when a channel is IDLE:

```sv
for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin
    if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin
        channel_serving_consumer[j] = 1;
        current_consumer[i] <= j;
        mem_read_valid[i] <= 1;
        mem_read_address[i] <= consumer_read_address[j];
        controller_state[i] <= READ_WAITING;
        break;  // First-come, first-served by index
    end
end
```

**The bias problem:** Consumer 0 (Core 0, Thread 0) always gets checked first. If all 8 threads request simultaneously, Core 0's threads get all 4 channels. Core 1 waits for a second round.

---

## Chapter 28: Stride Patterns

### Stride-1 (Sequential) — What matadd Does

```
Thread 0: addr 0, Thread 1: addr 1, Thread 2: addr 2, Thread 3: addr 3
```

In tiny-gpu: 4 separate requests. No optimization.

On a real GPU with coalescing: **1 transaction.** 4× bandwidth improvement.

### Stride-2 (Every Other Element)

```
Thread 0: addr 0, Thread 1: addr 2, Thread 2: addr 4, Thread 3: addr 6
```

In tiny-gpu: Still 4 separate requests. Same cost.

On a real GPU: Might span 2 cache lines → 2 transactions instead of 1. Still better than 4.

### Stride-N (Large Stride)

In tiny-gpu: Still 4 separate requests. All strides cost the same.

On a real GPU: **Worst case.** Each address hits a different cache line. No coalescing benefit. This is why GPU programming guides emphasize coalesced memory access.

---

## Chapter 29: Memory Banks

**tiny-gpu has no banks.** Memory is a flat Python array:

```python
class Memory:
    def __init__(self, dut, addr_bits, data_bits, channels, name):
        self.memory = [0] * (2**addr_bits)  # Just a flat array
```

Every channel can read any address instantly. No conflicts.

**Real GPU memory** (on-chip shared memory, GMEM) is organized into banks:

```
Bank 0: addresses 0, 16, 32, 48, ...
Bank 1: addresses 1, 17, 33, 49, ...
...
Bank 15: addresses 15, 31, 47, 63, ...
```

**Bank conflict** = two threads access the same bank in the same cycle. The bank serializes.

**Stride-1** → each thread hits a different bank → all parallel → fast.
**Stride-16** (assuming 16 banks) → all threads hit the same bank → serialized → slow.

---

## Chapter 30: The Handshake Protocol

The controller uses a valid/ready handshake per channel:

```
IDLE → READ_WAITING → READ_RELAYING → IDLE
```

The full round-trip for one memory read:

```
Cycle 1: LSU sets mem_read_valid=1
Cycle 2: Controller forwards to memory
Cycle 3: Memory responds
Cycle 4: Controller relays to LSU
Cycle 5: LSU grabs data, drops valid
Cycle 6: Controller frees channel
```

~4-6 cycles per memory access. And the core is stalled the entire time.

---

# Part VI: Mapping to Snapdragon 8 Elite

## Chapter 31: Snapdragon SoC Overview

A **Snapdragon SoC** (System on Chip) is an entire computer on one chip:

```
┌──────────────────────────────────────────────────────────────────┐
│                    Snapdragon 8 Elite SoC                        │
│                    TSMC N3E · 3nm · ~20B transistors              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   CPU         │  │   GPU         │  │   NPU                 │  │
│  │   Oryon Gen 2 │  │   Adreno 830  │  │   Hexagon             │  │
│  │   8 cores     │  │   3 slices    │  │   ~75 TOPS INT8       │  │
│  │   4.47 GHz    │  │   12 CU       │  │                       │  │
│  └──────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │   ISP         │  │   Modem       │  │   Connectivity        │  │
│  │   Spectra     │  │   X80 5G      │  │   FastConnect 7900    │  │
│  └──────────────┘  └──────────────┘  └────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              System-Level Cache (SLC) — 8 MB                 │ │
│  │         shared across CPU · GPU · NPU · ISP · modem          │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │           LPDDR5X Memory Controller — 84.8 GB/s              │ │
│  │           4 channels × 16-bit = 64-bit bus                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

The GPU is just **one block** on this chip. It shares the memory controller with the CPU, NPU, and everything else.

### Detailed Block Diagram — Data Paths and Roles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Snapdragon 8 Elite Detailed View                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Compute Subsystem                                 │    │
│  │                                                                      │    │
│  │  ┌──────────────────────┐  ┌────────────────────────────────────┐  │    │
│  │  │  CPU (Oryon Gen 2)   │  │  GPU (Adreno 830)                  │  │    │
│  │  │  8 cores, 4.47 GHz   │  │  3 slices × 4 CU = 12 CU total    │  │    │
│  │  │  • L1I: 192KB        │  │  • Each CU: 256 threads ready      │  │    │
│  │  │  • L1D: 128KB        │  │  • Each CU: ~256 KB register file  │  │    │
│  │  │  • L2: 24 MB shared  │  │  • L1 cache: per-CU               │  │    │
│  │  │                      │  │  • L2: 4 MB per slice             │  │    │
│  │  └──────────┬───────────┘  └──────────┬─────────────────────────┘  │    │
│  │             │                         │                             │    │
│  │             └─────────────┬───────────┘                             │    │
│  │                           │                                         │    │
│  └───────────────────────────┼─────────────────────────────────────────┘    │
│                              │                                               │
│  ┌──────────────────────────▼──────────────────────────────────────────┐    │
│  │           NoC Fabric (Network on Chip) — QoS-Aware                  │    │
│  │  • Priority routing per IP block (CPU > GPU > NPU > ISP)            │    │
│  │  • Crossbar arbitration with credit-based fairness                  │    │
│  │  • Connects to SLC and memory controller                            │    │
│  └──────────────────────┬───────────────────────────────────────────────┘   │
│                         │                                                    │
│  ┌──────────────────────▼──────────────────────────────────────────────┐    │
│  │          System-Level Cache (SLC) — 8 MB shared                     │    │
│  │  • Serves: CPU + GPU + NPU + ISP + Modem                            │    │
│  │  • Latency: ~30-40 cycles (much better than DRAM)                   │    │
│  └──────────────────────┬───────────────────────────────────────────────┘   │
│                         │                                                    │
│  ┌──────────────────────▼──────────────────────────────────────────────┐    │
│  │    LPDDR5X Memory Controller — 4 channels, ~84.8 GB/s peak          │    │
│  │  • Channel 0-3: each 16-bit × 9600 MT/s                             │    │
│  │  • Prioritizes real-time consumers (display, ISP)                    │    │
│  │  • Remaining bandwidth shared among CPU/GPU/NPU                      │    │
│  └──────────────────────┬───────────────────────────────────────────────┘   │
│                         │                                                    │
│                         ▼                                                    │
│              ┌─────────────────────┐                                         │
│              │  12 GB LPDDR5X DRAM │                                         │
│              │  (~100-150 ns lat)  │                                         │
│              └─────────────────────┘                                         │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │              NPU (Hexagon) — Separate Power Rail                     │    │
│  │  • Scalar + Vector + Tensor Accelerator                              │    │
│  │  • Can run at peak while GPU/CPU idle at low voltage                 │    │
│  │  • Independent DVFS for energy efficiency                             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### SoC vs Discrete GPU Comparison Table

| Metric | Snapdragon SoC (Integrated) | Discrete GPU (NVIDIA) |
|--------|-------|-------|
| **Power** | ~5W sustained | ~500-1000W TDP |
| **Memory** | Shared LPDDR5X (12 GB) | HBM3e (192 GB per GPU) |
| **Bandwidth** | ~84.8 GB/s shared | ~8 TB/s per GPU |
| **Compute** | 3.7 TFLOPS (GPU) + 75 TOPS (NPU) | 2,000+ TFLOPS (FP32) |
| **On-die cache** | 44+ MB total | 50+ MB per GPU |
| **CPU+GPU latency** | ~0 (same die) | ~10-100 μs (PCIe) |
| **Data movement** | No DMA copy needed | Host ↔ Device DMA overhead |
| **Scalability** | Single chip, up to 16GB | 8-16 GPUs per node |
| **Advantage** | Low power, zero-copy access | Massive parallelism, throughput |

The integrated design means the GPU doesn't need to copy data across PCIe — it's already on the same chip accessing shared memory through the NoC fabric. This zero-copy advantage is crucial for edge inference where latency matters.

---

## Chapter 31.5: Detailed Adreno 830 GPU Block Architecture

### Complete Single Compute Unit Block Diagram

Each of the 12 Adreno 830 Compute Units is a complete mini-GPU:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    Single Adreno 830 Compute Unit (CU)                     │
│                   Contains 64 fibers organized as 1 wave                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │              Command Processor Interface                             │  │
│  │  Receives: kernel launch, grid dimensions, program pointer          │  │
│  │  Feeds: workgroup ID, num_threads, base_SP_pointer                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │          Workgroup Scheduler per CU                                 │  │
│  │  • Holds up to 40 resident waves per CU                             │  │
│  │  • Selects which wave issues next instruction                       │  │
│  │  • Wakes sleeping waves when memory ready                           │  │
│  │  • Manages register allocation per wave                             │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │         Instruction Fetch & Decode Unit                             │  │
│  │  • Per-wave program counter (PC) + branch divergence stack          │  │
│  │  • Instruction cache (typical: 16-32 KB per CU)                     │  │
│  │  • Instruction decode → 6 micro-ops for current wave                │  │
│  │  • Latency: ~1-4 cycles if I-cache hit, ~20+ if miss                │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │  6 Shader Processors (SPs) — execute micro-ops in parallel          │  │
│  │                                                                      │  │
│  │  ┌─────────────┬─────────────┬─────────────┐                        │  │
│  │  │   SP[0]     │   SP[1]     │   SP[2]     │  First wave of 3 SPs  │  │
│  │  │ Fibers 0-10 │ Fibers 11-21│ Fibers 22-32│                        │  │
│  │  └─────────────┴─────────────┴─────────────┘                        │  │
│  │  ┌─────────────┬─────────────┬─────────────┐                        │  │
│  │  │   SP[3]     │   SP[4]     │   SP[5]     │  Second wave of 3 SPs │  │
│  │  │ Fibers 33-43│ Fibers 44-54│ Fibers 55-63│                        │  │
│  │  └─────────────┴─────────────┴─────────────┘                        │  │
│  │                                                                      │  │
│  │  Each SP contains:                                                  │  │
│  │  ┌────────────────────────────────────────┐                         │  │
│  │  │  Wave Scheduler (which 16 fibers next?)│                         │  │
│  │  │  Instruction Fetch/Decode              │                         │  │
│  │  │  Dual-ALU pipe (FP32 + INT32)          │                         │  │
│  │  │  MAD (Multiply-Add) unit                │                         │  │
│  │  │  Load/Store Unit (LSU)                  │                         │  │
│  │  │  Load/Store + Coalescing logic          │                         │  │
│  │  │  Register File (per-fiber: ~256 bytes)  │                         │  │
│  │  │  L1 Data Cache (per-SP or shared)       │                         │  │
│  │  └────────────────────────────────────────┘                         │  │
│  │                                                                      │  │
│  │  Execution Model:                                                   │  │
│  │  • Each SP handles 11-16 fibers sequentially (lockstep)             │  │
│  │  • 6 SPs in parallel = 64-96 fibers/cycle (depending on config)     │  │
│  │  • One ALU result per fiber per cycle (after pipelining)            │  │
│  │  • Total: 256+ FP32 ALUs effective from wavefront                   │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │         Shared Memory / LDS (64 KB per CU)                          │  │
│  │  • Accessible by all fibers in CU                                   │  │
│  │  • ~4-5 cycles latency (on-chip SRAM)                               │  │
│  │  • Thread synchronization via barriers                              │  │
│  │  • Not used for streaming loads; used for tile reuse                │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │         L1 Data Cache (32-64 KB per CU)                             │  │
│  │  • Cache line: 64 bytes                                             │  │
│  │  • Fully associative (for efficiency)                               │  │
│  │  • Write-back to L2                                                 │  │
│  │  • Latency: ~4 cycles hit, ~20+ on miss                             │  │
│  │  • Coalescing unit: merges adjacent addresses before fetch          │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │       L2 Cache (4 MB per slice, shared by 4 CUs)                    │  │
│  │  • Latency: ~20-40 cycles hit, ~100+ on miss                        │  │
│  │  • Write-back to SLC                                                │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │    SLC (System-Level Cache) & GMEM Bridge                           │  │
│  │  • 8 MB SLC shared across all IPs (CPU/GPU/NPU/ISP/Modem)          │  │
│  │  • Latency: ~30-40 cycles                                            │  │
│  └──────────────────────┬───────────────────────────────────────────────┘  │
│                         │                                                   │
│  ┌──────────────────────▼──────────────────────────────────────────────┐  │
│  │    Memory Controller & LPDDR5X                                      │  │
│  │  • 4 channels shared with CPU/NPU                                   │  │
│  │  • ~100-150 cycles latency to DRAM                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### tiny-gpu vs Adreno 830 Architecture Comparison

Each level shows exactly what tiny-gpu has vs doesn't have:

| Component | tiny-gpu | Adreno 830 per CU | Impact |
|-----------|----------|-------------------|--------|
| **Workgroup Scheduling** | 1 sequential state machine | 40 waves in-flight, dynamic scheduling | 60-95× better ALU utilization (hides latency) |
| **Instruction Fetch** | 1 channel × 2 cores, goes to ext mem | Per-wave I-cache hit → 1 cycle | 13× fewer ext mem fetches |
| **Register File** | 128 bytes total (4 threads × 32 bytes) | ~256 KB (all 64 fibers' state resident) | Enables wave multiplexing |
| **L1 Cache** | None (every access → memory) | 32-64 KB, write-back | 5-25× faster for repeated access |
| **L2 Cache** | None | 4 MB per slice | Coalescing window for L1 miss traffic |
| **SLC / Shared** | None | 8 MB | Reduces DRAM traffic by 10-20× for small models |
| **Coalescing** | No (8 separate requests) | Hardware: adjacent addrs → 1 transaction | 4-64× fewer memory transactions |
| **LSU Pipelining** | Serial (wait for response) | 16+ in-flight requests per CU | Hides latency while others compute |
| **ALU Count** | 8 total (4 per core) | 1,536 FP32 total (256 per CU) | 192× more ALUs |
| **Precision** | 8-bit fixed | FP32, FP16, INT32, INT8, INT4, Transcendentals | Native mixed-precision |
| **Wave Divergence** | Assumed convergent (TODO) | Full divergence support with stack | Handles complex control flow |

### Critical Bottleneck Removal: Memory Stalls

**tiny-gpu bottleneck in scheduler.sv:**
```sv
// The entire core WAITS (freezes all ALUs) until ANY thread finishes memory request
if (any_lsu_waiting) core_state <= WAIT;  // STALL
else core_state <= EXECUTE;
```

**Adreno solution in Wave Scheduler:**
```
Wave 0 requesting memory?  → Put it to sleep
Wave 1 ready to compute?   → Schedule it immediately
Wave 2 ready to compute?   → Schedule it immediately
Result: 6 SPs never idle while ANY wave is compute-ready
```

This **one architectural change** (wave scheduling) accounts for the majority of the 60-95× utilization improvement.

---

## Chapter 32: gpu.sv → Adreno 830

```
tiny-gpu                              Adreno 830
────────                              ──────────
NUM_CORES = 2                    →    3 slices × 4 CU = 12 Compute Units
THREADS_PER_BLOCK = 4            →    Waves of 64 fibers, 1000s per CU
DATA_MEM_NUM_CHANNELS = 4        →    4 LPDDR5X channels, 84.8 GB/s peak
PROGRAM_MEM_NUM_CHANNELS = 1     →    Per-CU instruction cache hierarchy
DATA_MEM_ADDR_BITS = 8 (256)     →    36+ bit address space (12 GB)
DATA_MEM_DATA_BITS = 8           →    32-bit (FP32) or 16-bit (FP16) native
```

---

## Chapter 33: dcr.sv → Command Processor

Where tiny-gpu has 1 register storing thread count, the Adreno 830's Command Processor has thousands of registers: grid dimensions, block dimensions, shader program pointers, constant buffer pointers, texture descriptors, render target config, and hundreds more.

The driver (`msm_kgsl` kernel module) writes to these through MMIO — conceptually identical to tiny-gpu's `device_control_write_enable` and `device_control_data`.

---

## Chapter 34: dispatch.sv → Workgroup Scheduler

The Adreno 830 has a **3-slice architecture** (first for Adreno). Individual slices can be power-gated:

```
GPU DVFS:
  1,200 MHz  ~0.85V  ~5.0W  → 3 slices active
    900 MHz  ~0.75V  ~2.5W  → 2-3 slices
    600 MHz  ~0.65V  ~1.2W  → 1-2 slices
    300 MHz  ~0.55V  ~0.3W  → 1 slice
```

tiny-gpu always uses both cores. The Adreno can shut down entire slices to save power.

---

## Chapter 35: core.sv → Compute Unit

```
tiny-gpu core                         Adreno 830 Compute Unit (CU)
─────────────                         ────────────────────────────────
1 fetcher                        →    Instruction fetch + I-cache
1 decoder                        →    Instruction decode + operand collector
1 scheduler (sequential FSM)     →    Wave scheduler (multiple waves in-flight)
4 ALUs (8-bit)                   →    256 FP32 ALUs + 512 FP16 ALUs per CU
4 LSUs                           →    Load/store unit + coalescing unit + L1 cache
4 register files (16×8-bit)      →    ~256 KB register file per CU
4 PCs                            →    Per-wave PC + divergence stack
No cache                         →    L1 cache + 4 MB slice L2 + 8 MB SLC
```

---

## Chapter 35.5: Detailed Memory Paths in core.sv vs Adreno CU

### tiny-gpu core.sv: Memory Request Path

The tiny-gpu core has a simple, sequential memory access pattern:

```sv
// From core.sv EXECUTE state
always @(posedge clk) begin
    case (core_state)
        EXECUTE: begin
            case (decoded_instruction_type)
                LDR: begin
                    // Thread sends load request
                    core_state <= WAIT;  // ← ENTIRE CORE STALLS
                    lsu_state[thread_id] <= REQUESTING;
                end
            endcase
        end
        WAIT: begin
            // Check if ANY thread is still waiting on LSU
            if (any_lsu_waiting) core_state <= WAIT;  // STALL
            else core_state <= EXECUTE;                // RESUME
        end
    endcase
end
```

**Memory Request Sequence (single thread, A[0]):**

```
Cycle 1:  [LDR executed]
          Thread 0 LSU: rs = 0x00000000 (compute address)

Cycle 2:  [WAIT state]
          Thread 0 LSU: mem_read_valid = 1, mem_read_address = 0x00
          ALL OTHER THREADS: FROZEN (core_state = WAIT)

Cycle 3:  [WAIT state]
          Memory Controller picks up request
          Sets mem_read_ready = 1, mem_read_data = A[0]

Cycle 4:  [WAIT state]
          Thread 0 LSU: mem_read_ready = 1 detected
          lsu_state[0] = DONE, lsu_out[0] = A[0]

Cycle 5:  [WAIT state continues]
          Checking: any_lsu_waiting?
          Only Thread 0 is done, so any_lsu_waiting still true
          core_state = WAIT (continues)

Cycle 6:  [EXECUTE resumes]
          All threads can now execute next instruction
          Data from A[0] available in registers
```

**Problem**: All 8 threads are frozen while 1 thread waits. Even if threads 1-7 have no dependency on the result, they cannot execute.

### Adreno 830 CU: Memory Request Path with Wave Scheduling

```sv
// Pseudo-code from Wave Scheduler
always @(posedge clk) begin
    // Check which waves are ready (not waiting on memory)
    for (int w = 0; w < RESIDENT_WAVES; w++) begin
        if (wave_state[w] == COMPUTE_READY) begin
            // Schedule this wave's next instruction
            selected_wave = w;
            // Issue to one of 6 SPs
            break;
        end
    end

    // If selected wave issues a memory request:
    if (selected_wave_issues_memory) begin
        wave_state[selected_wave] = MEMORY_WAIT;  // ← ONLY THIS WAVE SLEEPS
        // Other waves continue scheduling!
    end
end
```

**Memory Request Sequence (Wave 0, all 64 fibers issue LDR A[gid]):**

```
Cycle 1:  [Wave 0 executes LDR on SP0]
          All 64 fibers in Wave 0: rs = gid (thread ID)
          Coalescing unit examines addresses:
            Fiber 0:  A[0] → address 0
            Fiber 1:  A[1] → address 4
            Fiber 2:  A[2] → address 8
            ...
            Fiber 63: A[63] → address 252
          Adjacent addresses merged: {0-63} becomes 1 cache-line request

Cycle 2:  [Wave 0 sent to L1 cache, Wave 1 issued]
          Wave 0: memory_waiting = true, put to sleep
          Wave 1: selected, starts executing

Cycle 3:  [Wave 1 executing, Wave 0 in L1]
          L1 cache determines: address 0 not in cache (MISS)
          Sends request to L2 for cache line

Cycle 4:  [Wave 1 still executing, Wave 2 issued]
          Wave 1: executes ADD, MUL instructions
          Wave 0: L2 not ready yet

Cycle 5:  [Wave 2 executing, Wave 0 in L2]
          L2 cache hit? (unlikely first time)
          Sends request to SLC

Cycle 10: [Wave 3, 4, 5... all executing]
          While Waves 0-2 wait on different cache levels,
          Scheduler issues Waves 3-5 continuously
          All 6 SPs are busy

Cycle 20: [Wave 0 data returns from L2]
          L1 cache refilled with cache line {A[0-15]}
          Wave 0 state: MEMORY_WAIT → COMPUTE_READY

Cycle 21: [Wave 0 resumes execution]
          Scheduler picks Wave 0, resumes from memory_wait
          Next instruction uses data from registers
          Waves 1-5 continue executing
```

**Key Difference**: While Wave 0 waits 19 cycles, Waves 1-5 execute continuously on the 6 SPs. From the SP's perspective, they're never idle because a new wave always has work to do.

---

## Chapter 36: scheduler.sv → Wave Scheduler

This is where tiny-gpu and the Adreno diverge most dramatically.

tiny-gpu stalls the entire core when any thread waits for memory. The Adreno keeps **dozens of waves** resident per CU and switches between them:

```
tiny-gpu — serial stall:
─────────────────────────────────────────────────────────────────────────────
Thread block:  ▓▓▓▓ COMPUTE ▓▓▓▓  ░░░░ MEM WAIT ░░░░  ▓▓▓▓ COMPUTE ▓▓▓▓
ALU:           [busy]              [IDLE - wasted]      [busy]
─────────────────────────────────────────────────────────────────────────────

Adreno 830 — wave interleaving:
─────────────────────────────────────────────────────────────────────────────
Wave 0:  ▓▓▓▓ COMPUTE ▓▓▓▓  ░░░░ MEM WAIT... ░░░░░░░░        ...▓▓ COMPUTE
Wave 1:           ▓▓▓▓ COMPUTE ▓▓▓▓  ░░░░ MEM WAIT... ░░░░        ...▓▓
Wave 2:                    ▓▓▓▓ COMPUTE ▓▓▓▓  ░░ MEM WAIT ░░  ▓▓▓▓ COMPUTE
ALU:     [busy] [busy] [busy] [busy] [busy] [busy] [busy] [busy] [busy]
─────────────────────────────────────────────────────────────────────────────
Cost: must keep ALL resident wave register files live simultaneously
      → huge register files, most power-hungry component per area
```

ALU utilization comparison:

```
~120 total cycles  │████████████████████████████████████████████████████████████░░░░░░░░░░│
8 useful ALU-cyc   │█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
Utilization ≈ 0.8% ← not a bug; it's what happens without caches/coalescing/pipelining
```

This requires massive register files — every resident wave needs its own state kept alive.

---

## Chapter 37: controller.sv → Memory Hierarchy

tiny-gpu has one path: LSU → controller → external memory. Every access, every time.

The Adreno 830 has a deep hierarchy with multiple latency tiers:

```
Snapdragon 8 Elite                      NVIDIA B200

┌────────────────────┐                  ┌────────────────────┐
│  Tensor / scalar   │ ~1-4 cy          │  Tensor Cores +    │ ~1 cy
│  register files    │                  │  Register File     │
└────────┬───────────┘                  └────────┬───────────┘
         │                                       │
┌────────▼───────────┐                  ┌────────▼───────────┐
│  NPU scratchpad    │ ~5-10 cy         │  TMEM (Blackwell)  │ new in B200
│  (shared SRAM)     │                  │  (tensor memory)   │
└────────┬───────────┘                  └────────┬───────────┘
         │                                       │
┌────────▼───────────┐                  ┌────────▼───────────┐
│  GPU L1 cache      │ ~4 cy            │  SM L1 cache       │ ~30 cy
│  (per Compute Unit)│                  │  (per SM)          │
└────────┬───────────┘                  └────────┬───────────┘
         │                                       │
┌────────▼───────────┐                  ┌────────▼───────────┐
│  Slice L2 cache    │ ~20-40 cy        │  Shared L2 cache   │ ~200 cy
│  (per GPU slice)   │                  │  (~50 MB)          │
└────────┬───────────┘                  └────────┬───────────┘
         │                                       │
┌────────▼───────────┐                  ┌────────▼───────────┐
│  SLC (8 MB)        │ ~30-40 cy        │  HBM3e DRAM        │ ~400+ cy
│  shared: CPU+GPU   │                  │  192 GB, ~8 TB/s   │
│  +NPU+ISP          │                  │                    │
└────────┬───────────┘                  └────────────────────┘
         │
┌────────▼───────────┐
│  LPDDR5X DRAM      │ ~100-150 cy
│  12 GB, ~0.08 TB/s │
└────────────────────┘

Rule: every level exists to avoid the one below it.
```

Qualcomm describes **Adreno High Performance Memory (HPM)** providing 18 MB of dedicated memory cache and a "Tile Memory Heap" optimizing bandwidth and power.

---

## Chapter 38: alu.sv → Shader ALUs

tiny-gpu: 4 operations (ADD/SUB/MUL/DIV), 8-bit, 8 ALUs total.

Adreno 830: Full IEEE 754 FP32/FP16/INT32 pipelines, transcendentals (SIN, COS, EXP2, LOG2), texture filtering, ray tracing acceleration. 1,536 FP32 ALUs total. ~3.7 TFLOPS FP32, ~7.4 TFLOPS FP16.

---

## Chapter 39: lsu.sv → Load/Store + Coalescing

tiny-gpu: Each thread sends its own request. No merging. No caching.

Adreno 830: Coalescing unit examines all fibers' addresses within a wave. Consecutive addresses merged into one transaction. Write-back caching means stores hit L1 and the core moves on immediately.

---

## Chapter 39.5: Memory Hierarchy Deep Dive

### tiny-gpu Memory Path: Single Route, Every Access Identical Cost

```
Thread LSU sends request → Memory Controller → External Memory → Response → Thread reads
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                      SINGLE PATH, NO SHORTCUTS

Example: Load A[i] from address 0:
  Cycle 1: LSU_REQUESTING  (set mem_read_valid)
  Cycle 2: Controller picks up request
  Cycle 3: Memory responds (simulated instantly in Python)
  Cycle 4: Data back to LSU
  Cycle 5: Core can use data (UPDATE state)

  Minimum: 4-5 cycles
  Maximum: Infinite (no timeout in code)

  ALL 8 requests (A[0..7] and B[0..7] and C[0..7]) take identical time.
  NO CACHING, NO COALESCING, NO PARALLELISM.
```

**Memory Cycle Accounting for tiny-gpu Matrix Add:**
- First LDR (A[i]): ~4-5 cycles minimum per access
- Second LDR (B[i]): ~4-5 cycles (cache miss cost = memory miss cost)
- STR (C[i]): ~4-5 cycles
- × 8 threads × 3 memory ops = 24 separate transactions
- × 4-5 cycles = 96-120 cycles just for memory in a 13-instruction kernel

---

### Adreno 830 Memory Path: Hierarchy of Shortcuts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Coalescing Unit (per Shader Processor)             │
│  • Examine all 64 fiber addresses in current wave                       │
│  • Merge addresses 0, 1, 2, 3 → Single 64-byte cache-line request     │
│  • Merge addresses 8, 9, 10, 11 → Single cache-line request            │
│  Result: 8 requests become 2 transactions                               │
└────────────┬────────────────────────────────────────────────────────────┘
             │ 2 transactions
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              L1 Data Cache Check (per Shader Processor)                 │
│  • 64-byte line 0: in cache? → return in ~4 cycles                     │
│  • 64-byte line 8: not in cache? → go to L2                            │
│                                                                         │
│  First access (A[0-3]): MISS → L2                                      │
│  Second access (B[0-3]): HIT (same cache line) → ~4 cycles             │
│  Third access (C[0-3]): HIT (reuse fill?) → ~4 cycles                 │
└────────────┬────────────────────────────────────────────────────────────┘
             │ 1 transaction to L2 (per distinct cache line missed)
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              L2 Cache (per slice, shared by 4 CUs)                      │
│  • 4 MB cache, 64-byte lines                                            │
│  • If data was recently accessed by another CU, hit here                │
│  • ~20-40 cycles latency                                                │
│  First A load: miss → go to SLC                                         │
└────────────┬────────────────────────────────────────────────────────────┘
             │ 1 transaction to SLC (if L2 miss)
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              SLC (System-Level Cache) — 8 MB shared                     │
│  • All IP blocks (CPU, GPU, NPU, ISP) can hit here                     │
│  • If model weights cached by system, loads hit SLC: ~30-40 cyc         │
│  ~44 MB total on-chip SRAM (enough for small model weights)            │
└────────────┬────────────────────────────────────────────────────────────┘
             │ 1 transaction to DRAM (if SLC miss)
             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              LPDDR5X DRAM — 4 channels, ~84.8 GB/s                      │
│  • Latency: 60–200+ ns depending on access pattern:                    │
│      Page hit (tCL + tRCD ≈ 18ns each): ~60–80 ns                     │
│      Page miss (add tRP precharge):      ~80–120 ns                    │
│      Rank switching:                     ~120–200+ ns                  │
│      Plus QoS arbitration: if CPU is bursting, NPU waits               │
│  • At 1.2 GHz GPU clock: ~72–240+ cycles                              │
│  This is PAINFUL but unavoidable for weights not in on-chip memory      │
└─────────────────────────────────────────────────────────────────────────┘

Coalescing effectiveness:
  Without: 8 separate requests, each traverses full hierarchy
  With:    2 transactions (addresses 0-7 in same cache line → 1 transaction)
  Savings: 75% fewer external memory transactions
```

---

### Wave Scheduling Hiding Latency

The magic: while one wave waits for memory, scheduler switches to a different wave:

```
Timeline for 3 resident waves with memory stalls:

Wave 0:  ▓▓▓▓ COMPUTE ▓▓▓▓  ░░░░ L1 MISS (20 cyc) ░░░░░░░░░░░░░  ▓ COMPUTE
         Cycles: 1-4        Cycles: 5-24 (waiting)              Cycles: 25-26

Wave 1:               ▓▓▓▓ COMPUTE ▓▓▓▓  ░░░░ L1 HIT (4 cyc) ░░░░  ▓▓▓ COMPUTE
         Cycles:        5-8         9-28 (waiting)        Cycles: 29-31

Wave 2:                            ▓▓▓▓ COMPUTE ▓▓▓▓  ░░ L1 MISS ░░░░░ ▓ COMPUTE
         Cycles:                  9-12        13-32                 33-34

Actual execution:
Cyc 1-4:   Wave 0 compute (Wave 1, 2 waiting on scheduler)
Cyc 5-8:   Wave 1 compute (Wave 0 sent L1 miss to L2)
Cyc 9-12:  Wave 2 compute (Wave 1 sent L1 hit, continues)
Cyc 13-24: All compute-ready waves exhaust; Wave 0 still waiting
Cyc 25-26: Wave 0 resumes with data
...

Key insight: Waves with memory latency "hide" because other waves execute in the gaps.
SPs are almost never idle as long as ANY wave has work to do.
```

---

### Performance Numbers: Concrete Comparison

For the matadd kernel (8 elements, 3 loads + 1 store + 1 add per element):

```
tiny-gpu:
─────────
  First load (L1):  ~4-5 cycles (simulated memory)
  Second load (L2): ~4-5 cycles (no cache, same cost)
  Store:            ~4-5 cycles
  ALU ops:          ~7 cycles per instruction overhead
  Total per thread: ~120-130 cycles

  Throughput: 1 thread block / ~120 cycles
            = ~8 pixels/cycle (for 8 threads)

Adreno 830 (single wave, no other waves):
──────────────────────────────────────────
  First load (L1 miss + coalescing):
    Coalescing: merge 8 requests → 1 (or 2 depending on alignment)
    L1 miss → L2: ~20 cycles
  Second load (L1 hit):
    Same cache line, ~4 cycles
  Store (write-back):
    Hits L1, returns immediately: ~2 cycles
  ALU ops (pipelined):
    ~1 cycle per instruction
  Total: ~41 cycles (5× faster)

Adreno with wave scheduling (realistic workload with 10-20 waves):
──────────────────────────────────────────────────────────────────
  While Wave 0 waits on L1 miss (20 cyc),
  Scheduler immediately switches to Wave 1, 2, 3, etc.
  Visible latency: ~20 cycles but parallelized across 10 waves
  Effective latency per wave: ~2 cycles (hidden behind other waves)
  Total per wave: ~20 visible cycles (ALU time), other waves compute in parallel
  Throughput: 10-20 thread blocks / ~20 visible cycles
            = 30-60 pixels/cycle (same SPs, 4-7× better utilization)
```

**Summary Table:**

| Metric | tiny-gpu | Adreno 830 | Ratio |
|--------|----------|-----------|-------|
| First load latency | 4-5 cyc | 20 cyc (L1 miss) | 4-5× SLOWER |
| Second load latency | 4-5 cyc | 4 cyc (L1 hit) | 1.2-1.25× faster |
| Per-thread throughput | 8/120 = 0.067 threads/cyc | 10-20 waves in parallel | 6-8× faster per thread |
| Total throughput | 8 threads / 120 cyc | 640-1280 threads / 20 visible cyc | 16-100× more threads |
| Memory transactions (per kernel) | 24 | 3-4 (coalesced) | 6-8× fewer |
| **Overall speedup** | 1× baseline | **6-8× per thread + 16× more threads = 100× more throughput** | 100× |

The 100× improvement comes from:
- 6-8× faster per-thread execution (caching + coalescing)
- 16× more threads in parallel (wave scheduling)
- Overlap that hides latency within the wave scheduling

---

## Chapter 40: The Power Dimension

tiny-gpu has no concept of power management. Both cores always run. All ALUs always clocked.

The Snapdragon 8 Elite has independent power rails from the PM8750 PMIC:

```
VddCx    → CPU digital logic
VddGfx   → GPU (per-slice power gating)
VddNPU   → NPU tensor accelerator (INDEPENDENT DVFS)
VddMx    → Memory logic
VddMss   → Modem
VddCam   → ISP/camera
```

The NPU tensor accelerator having its own power rail means it can run at max while scalar/vector cores idle at low voltage during pure matrix-multiply workloads.

Sustained thermal budget in the Galaxy S25+: ~4-5W. Peak burst: ~8-10W before throttling.

---

## Chapter 41: The Software Stack

```
tiny-gpu                              Snapdragon
────────                              ──────────
Python test (setup.py)           →    QNN / AI Engine Direct (unified API)
Hand-encoded binary              →    Qualcomm shader compiler
program_memory.load(program)     →    Driver DMA of shader binary
data_memory.load(data)           →    clEnqueueWriteBuffer() / vkCmdCopyBuffer()
dut.device_control_data = 8      →    Driver writes to CP registers
dut.start = 1                    →    Driver kicks Command Processor
while dut.done != 1              →    clFinish() / vkWaitForFences()
data_memory.memory[16]           →    clEnqueueReadBuffer() / vkMapMemory()
```

Key software note: Android's NNAPI is deprecated in Android 15. The ecosystem is shifting toward LiteRT's QNN accelerator integration and Qualcomm's AI Engine Direct (QNN) with ahead-of-time and on-device compilation for Hexagon/HTP backends.

---

## Chapter 41.5: Complete Software Stack Path

### tiny-gpu: Human-Encoded Binary → Direct Execution

```python
# File: test/helpers/setup.py

# Step 1: Human hand-encodes binary instruction
#   16-bit instruction: [opcode=4b, rd=4b, rs=4b, rt=4b]
program = [
    0x5000 + 0x00D0 + 0x000E,  # MUL R0, %blockIdx, %blockDim
    0x3000 + 0x0000 + 0x000F,  # ADD R0, R0, %threadIdx
    # ... more hand-encoded instructions ...
]

# Step 2: Load directly into simulated GPU memory (no compilation, no driver)
program_memory.load(program)
data_memory.load(data)

# Step 3: Write thread count to DCR (device control register)
dut.device_control_write_enable.value = 1
dut.device_control_data.value = 8

# Step 4: Kick GPU (no OS, no driver, direct HW access in simulation)
dut.start.value = 1

# Step 5: Poll for done (active wait, busy-spin in testbench)
while dut.done != 1:
    pass

# Step 6: Read results (direct memory read)
results = data_memory.memory[16:24]
```

**Key characteristics:**
- No compiler (hand-written binary)
- No driver (direct simulation)
- No runtime overhead
- Single binary format (tiny-gpu ISA only)
- Total execution time: ~120 cycles in simulation, instant in real simulation

---

### Snapdragon: Framework → Compiler → Driver → GPU

```
┌──────────────────────────────────────────────────────────────────┐
│  Application (TensorFlow Lite, PyTorch)                          │
│                                                                  │
│  model = tf.lite.Interpreter("quantized_model.tflite")         │
│  output = model.predict(input_data)                            │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  LiteRT Framework + QNN Accelerator Plugin                       │
│  • Converts model graph to QNN ops                               │
│  • Selects hardware backend (HTP for Hexagon, GPU for Adreno)   │
│  • Example: Conv2D layer → HTP tensor microkernal               │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  QNN SDK / Compiler Pipeline (Qualcomm proprietary)             │
│                                                                  │
│  1. Graph Optimization                                           │
│     • Operator fusion (Conv+ReLU → single kernel)               │
│     • Layer tiling for on-chip SRAM reuse                        │
│     • Quantization-aware fusion (INT4 packing)                   │
│                                                                  │
│  2. Backend Code Generation                                      │
│     • Hexagon target: HVX scalar code + tensor accelerator code │
│     • Adreno target: RDNA-style compute shader + coalescing      │
│     • Generate microkernel binaries per layer                    │
│                                                                  │
│  3. Memory Planning                                              │
│     • Allocate on-chip SRAM (44+ MB total)                      │
│     • Assign tiles to SRAM for micro-tile inferencing            │
│     • Generate DMA descriptors for prefetching                   │
│                                                                  │
│  4. Emit Driver IR                                               │
│     • Sequence of GPU/NPU commands in driver format              │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  GPU Driver (msm_kgsl kernel module in Linux, Qualcomm's in QNX) │
│                                                                  │
│  1. Memory Management                                            │
│     • Allocate GPU-visible buffers (IOMMU mapping)               │
│     • DMA copy model weights to GPU memory                       │
│     • DMA copy input activations                                 │
│                                                                  │
│  2. Command Buffer Preparation                                   │
│     • Create command queue (circular buffer)                     │
│     • Emit GPU commands (CP packets)                             │
│       - Draw/compute dispatch: grid dimension, workgroup size    │
│       - Set shader binary pointer (GPU memory address)           │
│       - Bind constant buffers, texture descriptors              │
│       - Issue kernel dispatch                                    │
│                                                                  │
│  3. Synchronization                                              │
│     • Write fence command (memory barrier)                       │
│     • GPU writes fence value when kernel done                    │
│     • Driver polls or awaits IRQ (interrupt)                     │
│                                                                  │
│  4. Power Management                                             │
│     • DVFS: Request GPU frequency based on workload              │
│     • Slice power-gating: Disable unused slices                  │
│     • Clock gating: Idle stages within core                      │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Adreno 830 GPU Command Processor (CP)                           │
│                                                                  │
│  1. Decode CP packets (driver-issued commands)                   │
│     • LOAD shader binary to I-cache                              │
│     • SET grid dimensions (num workgroups, threads per group)   │
│     • SET register file pointers (const buffers, texture cache) │
│     • ISSUE dispatch                                             │
│                                                                  │
│  2. Dispatch Logic (maps to dispatch.sv in tiny-gpu)             │
│     • Calculate total workgroups                                 │
│     • Assign workgroups to CU slices round-robin                 │
│     • Load balance across 3 slices                               │
│                                                                  │
│  3. Send to Workgroup Scheduler                                  │
│     • Each CU receives workgroups                                │
│     • CU maintains queue of pending workgroups                   │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Adreno 830 Compute Unit (one of 12)                             │
│                                                                  │
│  1. Workgroup Scheduler (maps to scheduler.sv in tiny-gpu)       │
│     • Allocate registers for incoming wave                       │
│     • Set %blockIdx, %blockDim, %threadIdx read-only regs        │
│     • Create wave with 64 fibers                                 │
│     • Push to ready-to-issue queue                               │
│                                                                  │
│  2. Wave Scheduler (improvement over tiny-gpu's WAIT stall)     │
│     • Track which of 40 resident waves can issue next            │
│     • Select wave with earliest ready instruction                │
│     • Issue micro-ops to 6 Shader Processors                     │
│                                                                  │
│  3. Instruction Execution (Shader Processors)                    │
│     • Fetch/decode/execute loop identical to tiny-gpu core.sv   │
│     • Each SP handles 11-16 fibers sequentially                  │
│     • ALU, LSU, register file per fiber                          │
│                                                                  │
│  4. Memory Hierarchy (controller.sv + caching)                   │
│     • Coalescing: 64 fiber addresses → fewer transactions        │
│     • L1 cache: per-SP, 4-cycle hit                              │
│     • L2 cache: per-slice, 20-40 cycle hit                       │
│     • SLC: shared, 30-40 cycle hit                               │
│     • LPDDR5X: 100-150 cycles                                    │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  Memory Hierarchy (physical hardware)                            │
│                                                                  │
│  • Register file (per-fiber): ~4 bytes latency                   │
│  • L1 cache: ~4 cycles                                           │
│  • L2 cache: ~20-40 cycles                                       │
│  • SLC (8 MB): ~30-40 cycles                                     │
│  • LPDDR5X DRAM: ~100-150 cycles                                 │
│                                                                  │
│  Note: This is the SAME hierarchy as tiny-gpu's controller.sv   │
│        but with actual caching instead of simulation             │
└────────────────────────────────────────────────────────────────────┘
```

### Mapping Table: tiny-gpu → Snapdragon

| tiny-gpu Stage | Snapdragon Equivalent | Location |
|---|---|---|
| **setup.py:** hand-encode binary | QNN compiler generates shader binary | Compile time on CPU |
| **program_memory.load()** | Driver DMA loads binary to GPU I-cache | Kernel launch in driver |
| **data_memory.load()** | clEnqueueWriteBuffer() / vkCmdCopyBuffer() | Host → GPU memory via DMA |
| **dut.device_control_data = 8** | Driver writes to CP registers: grid dims | msm_kgsl register write |
| **dut.start = 1** | Driver issues CP_DRAW_INDIRECT or CP_INDIRECT_BUFFER | CP command queue submission |
| **while dut.done != 1** | clFinish() / vkWaitForFences() | Kernel completion wait (poll or IRQ) |
| **data_memory.memory[16:24]** | clEnqueueReadBuffer() / vkMapMemory() | GPU → Host memory via DMA |

### Key Differences in Software Stack

| Aspect | tiny-gpu | Snapdragon |
|--------|----------|-----------|
| **Compilation** | None (hand binary) | Proprietary QNN compiler |
| **Driver** | Simulation (no real driver) | Linux kernel module (msm_kgsl) |
| **Memory Model** | Flat 256-byte array | 12 GB LPDDR5X with MMU/IOMMU |
| **Synchronization** | Busy-wait (polling) | Interrupts or event-based wait |
| **Power Management** | None (always on) | DVFS + slice power gating |
| **Precision** | 8-bit fixed | INT4, INT8, FP16, FP32, mixed |
| **Latency** | ~120 cycles (entire kernel) | ~10-100 μs (host overhead) |
| **Throughput** | Limited by 8 threads | Limited by 1000s of threads × bandwidth |

---



# Part VII: Inference Side-by-Side

## Chapter 42: Setup Phase — Loading the Kernel

### The Computation: Matrix Addition

```
A = [0, 1, 2, 3, 4, 5, 6, 7]
B = [0, 1, 2, 3, 4, 5, 6, 7]
C = A + B = [0, 2, 4, 6, 8, 10, 12, 14]
```

**The kernel (13 instructions, 3 memory accesses):**

```asm
MUL R0, %blockIdx, %blockDim     ; i = blockIdx * blockDim + threadIdx
ADD R0, R0, %threadIdx
CONST R1, #0                      ; baseA = 0
CONST R2, #8                      ; baseB = 8
CONST R3, #16                     ; baseC = 16
ADD R4, R1, R0                    ; addr_A = baseA + i
LDR R4, R4                        ; R4 = A[i] ← MEMORY ACCESS 1
ADD R5, R2, R0                    ; addr_B = baseB + i
LDR R5, R5                        ; R5 = B[i] ← MEMORY ACCESS 2
ADD R6, R4, R5                    ; R6 = A[i] + B[i]
ADD R7, R3, R0                    ; addr_C = baseC + i
STR R7, R6                        ; C[i] = R6 ← MEMORY ACCESS 3
RET
```

### tiny-gpu Setup

```python
program_memory.load(program)    # 13 × 16-bit instructions → addresses 0-12
data_memory.load(data)          # 16 × 8-bit values → addresses 0-15
dut.device_control_write_enable.value = 1
dut.device_control_data.value = 8   # thread_count = 8
dut.start.value = 1             # GO
```

**Memory layout after setup:**

```
PROGRAM MEMORY (16-bit per row):
  Addr 0:  MUL R0, %blockIdx, %blockDim
  Addr 1:  ADD R0, R0, %threadIdx
  ...
  Addr 12: RET

DATA MEMORY (8-bit per row):
  Addr 0-7:    [0, 1, 2, 3, 4, 5, 6, 7]      ← Matrix A
  Addr 8-15:   [0, 1, 2, 3, 4, 5, 6, 7]      ← Matrix B
  Addr 16-23:  [?, ?, ?, ?, ?, ?, ?, ?]      ← Matrix C (output)
```

### Snapdragon 8 Elite Setup

```
Application (Java/NDK)
  │  OpenCL: clEnqueueWriteBuffer(A), clEnqueueWriteBuffer(B)
  │  OpenCL: clEnqueueNDRangeKernel(matadd, global_size=8)
  ▼
GPU Driver (msm_kgsl)
  │  1. Allocate GPU-visible memory in LPDDR5X
  │  2. DMA copy A, B → GPU memory
  │  3. Compile kernel → Adreno ISA binary
  │  4. Write shader binary pointer to Command Processor registers
  │  5. Write grid dimensions (8 threads) to CP registers
  │  6. Write buffer pointers (A, B, C addresses) to CP registers
  │  7. Kick Command Processor via ring buffer doorbell
  ▼
Command Processor + Workgroup Scheduler
  │  Ready to execute
```

**Key difference #1:** tiny-gpu loads into 256 bytes of flat memory. Snapdragon loads into 12 GB LPDDR5X through DMA, NoC, SLC, and memory controller.

---

## Chapter 43: Dispatch Phase — Distributing Work

### tiny-gpu Dispatcher

```sv
total_blocks = (8 + 4 - 1) / 4 = 2 blocks
```

```
Block 0 → Core 0: blockIdx=0, threads 0-3
Block 1 → Core 1: blockIdx=1, threads 0-3
```

From `dispatch.sv`, both cores start simultaneously:

```sv
core_start[0] <= 1; core_block_id[0] <= 0; core_thread_count[0] <= 4;
core_start[1] <= 1; core_block_id[1] <= 1; core_thread_count[1] <= 4;
```

Each thread gets hardwired identity from `registers.sv`:

```sv
registers[13] <= block_id;          // %blockIdx
registers[14] <= THREADS_PER_BLOCK; // %blockDim = 4
registers[15] <= THREAD_ID;         // %threadIdx = 0,1,2,3
```

### Snapdragon 8 Elite Scheduler

```
8 threads, 1 workgroup → 1 wave of 8 fibers

Slice 0:
  CU 0: Workgroup 0 → wave (8 fibers active, 56 masked)
  CU 1-3: idle

Slices 1-2: power-gated (VddGfx lowered, clocks stopped)
```

**Key difference #2:** Snapdragon power-gates 2 entire slices for this tiny workload.

**Key difference #3:** All 8 threads in one wave on one CU (no inter-core contention).

---

## Chapter 44: Computing the Thread Index

### tiny-gpu — Instruction 0: `MUL R0, %blockIdx, %blockDim`

**Detailed cycle trace for Core 0:**

```
Cycle 1: core_state = IDLE → start=1 → FETCH
         Fetcher: IDLE → FETCHING, mem_read_valid=1, address=0

Cycle 2: Fetcher: FETCHING (request to program memory)
         Controller: IDLE → picks Core 0 (fixed-priority scan)
         (Core 1 BLOCKED — only 1 channel!)

Cycle 3: Controller: memory responds → READ_RELAYING
         Fetcher: mem_read_ready=1 → FETCHED

Cycle 4: core_state = FETCH → DECODE
         Decoder: instruction = 0101_0000_1101_1110
           opcode = 0101 (MUL)
           rd = 0000 (R0)
           rs = 1101 (R13 = %blockIdx)
           rt = 1110 (R14 = %blockDim)

Cycle 5: All 4 register files read rs and rt:
         Thread 0: rs=0 (block 0), rt=4 (blockDim)
         Thread 1: rs=0, rt=4
         Thread 2: rs=0, rt=4
         Thread 3: rs=0, rt=4

Cycle 6: core_state = REQUEST → WAIT
         No LSUs active → any_lsu_waiting=0 → immediate pass

Cycle 7: core_state = WAIT → EXECUTE
         All 4 ALUs compute: 0 * 4 = 0

Cycle 8: core_state = EXECUTE → UPDATE
         All 4 register files write: R0 = 0
         PC advances: 0 → 1
         core_state = UPDATE → FETCH (next instruction)
```

**Core 1 stalled 2 cycles waiting for program memory.**

### tiny-gpu — Instruction 1: `ADD R0, R0, %threadIdx`

```
Core 0: Cycles 9-16: FETCH (delayed waiting for Core 1 to release channel)
                     ...
Core 1: Cycles 11-18: FETCH (finally gets program memory channel)
```

**~16 cycles total for 2 instructions, both cores.** Program memory contention delays Core 1.

### Snapdragon 8 Elite — Two Instructions

```
Cycle 1-4: Fetch MUL (I-cache MISS, external fetch)
Cycle 5:   Decode MUL
Cycle 6:   Execute MUL (all 8 fibers compute 0*8=0 simultaneously)
Cycle 7:   Fetch ADD (I-cache HIT, 1 cycle due to prefetch)
Cycle 8:   Decode ADD
Cycle 9:   Execute ADD
           Fiber 0: 0 + 0 = 0
           Fiber 1: 0 + 1 = 1
           Fiber 2: 0 + 2 = 2
           ...
           Fiber 7: 0 + 7 = 7
```

**~9 cycles for 2 instructions, all 8 threads. I-cache and pipelining eliminate contention.**

**Key difference #4:** Instruction cache. L0 I-cache holds instructions after first fetch. Sequential fetches are 1 cycle. tiny-gpu fetches from external memory every time through a 1-channel controller.

**Key difference #5:** Pipelining. Adreno overlaps fetch/decode/execute. tiny-gpu completes one stage before starting the next.

---

## Chapter 45: The First Memory Load ⚡

```asm
ADD R4, R1, R0    ; R4 = baseA + i = 0 + i
LDR R4, R4        ; R4 = A[i] ← LOAD A[THREAD_ID]
```

### tiny-gpu — Address Calculation

Address computation takes ~7-8 cycles. Each thread computes:

```
Core 0:                      Core 1:
  Thread 0: R4 = 0 + 0 = 0    Thread 0: R4 = 0 + 4 = 4
  Thread 1: R4 = 0 + 1 = 1    Thread 1: R4 = 0 + 5 = 5
  Thread 2: R4 = 0 + 2 = 2    Thread 2: R4 = 0 + 6 = 6
  Thread 3: R4 = 0 + 3 = 3    Thread 3: R4 = 0 + 7 = 7
```

### tiny-gpu — The Load: `LDR R4, R4`

**This is the architectural bottleneck.**

```
Cycle L+0: core_state → REQUEST
           All 4 LSUs in Core 0 read rs (R4 values 0,1,2,3)
           All 4 LSUs in Core 1 read rs (R4 values 4,5,6,7)

Cycle L+1: LSU REQUESTING
           Core 0: 4 LSUs transition IDLE → REQUESTING
           Core 1: 4 LSUs transition IDLE → REQUESTING

           mem_read_valid = 1 for all 8 threads
           mem_read_address = [0,1,2,3,4,5,6,7]

Cycle L+2: Controller ARBITRATION (fixed-priority scan in controller.sv)
           for (int j = 0; j < 8; j++) {
               if (consumer_read_valid[j] && !channel_serving_consumer[j]) {
                   channel_serving_consumer[j] = 1;
                   break;  // ← Consumers 0-3 (Core 0) always first
               }
           }

           Channel 0 → Consumer 0 (Core 0, Thread 0, addr 0)
           Channel 1 → Consumer 1 (Core 0, Thread 1, addr 1)
           Channel 2 → Consumer 2 (Core 0, Thread 2, addr 2)
           Channel 3 → Consumer 3 (Core 0, Thread 3, addr 3)

           Consumers 4-7 (Core 1): BLOCKED
           channel_serving_consumer = 0b00001111

Cycle L+3: Controller channels in READ_WAITING
           External memory (Python simulator) responds:
             Channel 0: data = 0 (A[0])
             Channel 1: data = 1 (A[1])
             Channel 2: data = 2 (A[2])
             Channel 3: data = 3 (A[3])

           State → READ_RELAYING

Cycle L+4: Core 0 LSUs receive data
           Thread 0: lsu_out = 0, state → DONE
           Thread 1: lsu_out = 1, state → DONE
           Thread 2: lsu_out = 2, state → DONE
           Thread 3: lsu_out = 3, state → DONE

           mem_read_valid drops, channels freed
           channel_serving_consumer = 0b00000000

Cycle L+5: Scheduler checks: any_lsu_waiting? NO → EXECUTE

Cycle L+5 (same): Controller NOW picks up Core 1 requests
           Channel 0 → Consumer 4 (Core 1, Thread 0, addr 4)
           Channel 1 → Consumer 5 (Core 1, Thread 1, addr 5)
           Channel 2 → Consumer 6 (Core 1, Thread 2, addr 6)
           Channel 3 → Consumer 7 (Core 1, Thread 3, addr 7)

Cycle L+6: Memory responds with Core 1 data [4,5,6,7]

Cycle L+7: Core 1 LSUs receive data, state → DONE
           Scheduler: any_lsu_waiting? NO → EXECUTE
```

**Timeline visualization:**

```
Cycle:      L  L1  L2  L3  L4  L5  L6  L7
Core 0:     REQ  ──  ──  DATA ── DONE EX
Core 1:     REQ  WAIT WAIT WAIT REQ ── DATA DONE
ALU 0-3:    [idle] [idle] [idle] [idle] EXECUTE
ALU 4-7:    [idle] [idle] [idle] [idle] [idle] [idle] EXECUTE

Core 1 starved for 3 cycles (L2-L4)
```

**Problems:**
1. **Starvation:** Fixed-priority arbiter in controller.sv always serves Core 0 first
2. **ALU idle:** 8 ALUs × 3 cycles wasted = 24 ALU-cycles lost
3. **No coalescing:** 8 separate requests (0,1,2,3,4,5,6,7) instead of 1 merged request

### Snapdragon 8 Elite — The Load: `LDR R4, R4`

All 8 fibers execute `LDR R4, R4` simultaneously:

```
Cycle M+0: Wave scheduler issues LDR to all 8 fibers
           Fiber addresses: [0, 1, 2, 3, 4, 5, 6, 7]

           COALESCING UNIT examines all 8 addresses:
             ┌─ Fiber 0: addr 0 ─┐
             ├─ Fiber 1: addr 1  │
             ├─ Fiber 2: addr 2  │ All within one 64-byte cache line!
             ├─ Fiber 3: addr 3  ├─→ COALESCED into 1 transaction
             ├─ Fiber 4: addr 4  │  Request: "read 8 bytes starting at addr 0"
             ├─ Fiber 5: addr 5  │
             ├─ Fiber 6: addr 6  │
             └─ Fiber 7: addr 7 ─┘

           8 requests → 1 memory transaction

Cycle M+1: L1 cache check: MISS (first access to A[])
           → Forward to Slice L2

Cycle M+2: L2 cache check: MISS
           → Forward to SLC

Cycle M+3: SLC check: MISS
           → Forward to LPDDR5X memory controller

Cycle M+4 to M+20: DRAM latency (~100-150 ns at 1.2 GHz ≈ ~100+ cycles)
           For this simulation, assume ~20 cycles due to system overhead

           Data flows: DRAM → Memory NoC → SLC → L2 → L1

Cycle M+21: Data arrives at CU
            Coalescing unit distributes bytes to each fiber:
              Fiber 0 → R4 = 0
              Fiber 1 → R4 = 1
              ...
              Fiber 7 → R4 = 7

            Data cached in L1, L2, SLC simultaneously

Cycle M+22: Wave state: MEMORY_WAIT → COMPUTE_READY
            Scheduler can now issue next instruction to Wave 0
```

**Key insight:** Unlike tiny-gpu, the Adreno has ONLY ONE load in-flight (because coalescing merged 8 into 1). **But** if there were other resident waves, the scheduler would switch to them during cycles M+4-M+20. ALUs would never stall.

---

## Chapter 46: The Second Memory Load — Cache Payoff

### tiny-gpu — `ADD R5, R2, R0` then `LDR R5, R5`

Same pattern as before: address calc (~7-8 cycles), then load (same 7-8 cycles with Core 1 starvation).

**No caching:** Addresses [8,9,10,11,12,13,14,15] are completely different from [0,1,2,3,4,5,6,7]. Same cost as first load. Core 1 starves again.

```
First load:   8 requests,4 channels, 3 cycles starvation
Second load:  8 requests, 4 channels, 3 cycles starvation
Total:        16 cycles + overhead
```

### Snapdragon 8 Elite — Second Load: `LDR R5, R5`

```
Cycle B+0: Wave scheduler issues LDR for addresses [8,9,10,11,12,13,14,15]

           Coalescing: 8 addresses → 1 transaction

Cycle B+1: L1 cache check: **HIT!**

           Why? The first load's cache-line fill brought in 64 bytes
           from address 0. That includes addresses 0-63. The second
           load requests addresses 8-15, which are within that same
           64-byte line already cached.

           L1 hit → return data in ~4 cycles

Cycle B+5: Data arrives (read from L1 cache, not DRAM)
           Fibers receive: R5 = [8,9,10,11,12,13,14,15]
```

**Performance comparison:**

```
                    First LDR       Second LDR
tiny-gpu:           ~7-8 cycles     ~7-8 cycles    (no cache, same cost)
Adreno 830:         ~20 cycles      ~4 cycles      (L1 hit, 5× faster)
```

**Why L1 hit:** Cache line granularity. Loading 8 bytes at address 0 fetches 64 bytes (cache line size). The second load at address 8 is already resident.

---

## Chapter 47: The Computation

### tiny-gpu: `ADD R6, R4, R5`

```asm
ADD R6, R4, R5    ; R6 = A[i] + B[i]
```

~7-8 cycles through FETCH→DECODE→REQUEST→WAIT→EXECUTE→UPDATE. The actual ALU operation is 1 cycle. The other 6-7 are overhead (instruction fetch, decode, register read/write).

### Snapdragon: Same instruction

I-cache hit (sequential access). Decode → Execute pipelined (~1 cycle). The actual ADD is free.

**Key insight:** The computation is trivial on both systems. All the performance difference comes from memory and instruction delivery.

---

## Chapter 48: The Store

### tiny-gpu: `ADD R7, R3, R0` then `STR R7, R6`

Address calc + store: Same pattern. 8 threads send 8 store requests. Core 1 waits. Same starvation.

Memory write must wait for acknowledgment from external memory before core can move on.

### Snapdragon: Same instructions

**Write-back caching:** Store hits L1 cache. Dirty bit set. Core moves on immediately. Cache controller asynchronously flushes to DRAM later.

Store: ~1-2 cycles (write to L1). Actual DRAM write deferred.

---

## Chapter 49: Kernel Complete

### tiny-gpu

Each core signals `done` to the dispatcher. When `blocks_done == 2`, test detects `dut.done == 1`.

### Snapdragon

GPU writes fence value to memory. Command Processor triggers interrupt. Driver wakes waiting thread. Application's `clFinish()` returns.

---

## Chapter 50: Complete Timeline Comparison

### tiny-gpu Detailed Breakdown

```
Setup + Dispatch:                     ~5 cycles
Instructions 0-4 (compute):           ~40 cycles (index calc + constants)
Instruction 5 (addr A):               ~8 cycles
Instruction 6 (LDR A[i]):             ~8 cycles (Core 0 finishes early,
                                                  Core 1 stalled 3 cycles)
Instruction 7 (addr B):               ~8 cycles
Instruction 8 (LDR B[i]):             ~8 cycles (Core 1 stalled again)
Instruction 9 (ADD):                  ~8 cycles
Instruction 10 (addr C):              ~8 cycles
Instruction 11 (STR):                 ~8 cycles (Core 1 stalled)
Instruction 12 (RET):                 ~4 cycles
─────────────────────────────────────────────────
TOTAL:                                ~113 cycles

Useful ALU work: 3 additions (one per thread pair)
ALU availability: 8 ALUs × 113 cycles = 904 ALU-cycles
Actual use: ~8 ALU-cycles
UTILIZATION: 8 / 904 ≈ 0.9%
```

### Snapdragon Detailed Breakdown

```
Setup + Dispatch:                     ~3 cycles
Instructions 0-4 (compute):           ~6 cycles (pipelined, I-cache after first)
Instruction 5 (addr A):               ~1 cycle
Instruction 6 (LDR A[i]):             ~20 cycles (first DRAM access, uncached)
Instruction 7 (addr B):               ~1 cycle
Instruction 8 (LDR B[i]):             ~4 cycles (L1 HIT from previous line fill)
Instruction 9 (ADD):                  ~1 cycle
Instruction 10 (addr C):              ~1 cycle
Instruction 11 (STR):                 ~2 cycles (write-back to L1, DRAM async)
Instruction 12 (RET):                 ~1 cycle
─────────────────────────────────────────────────
TOTAL:                                ~40 cycles (single wave, no scheduling)

With wave scheduling (10+ waves resident):
  Real latency = still ~20 cycles (DRAM)
  But HIDDEN: other waves execute during those 20 cycles
  Visible latency per wave: ~5-8 cycles

ALU availability: 256 per CU
Instruction pipelining: ~1 cycle per instruction
Total useful: ~20 cycles
UTILIZATION: ~60-95% (depends on wave count)
```

### Summary Table

| Metric | tiny-gpu | Adreno 830 | Ratio |
|--------|----------|-----------|-------|
| **Total cycles (this kernel)** | ~113 | ~40 | 2.8× |
| **Effective cycles (real load)** | ~113 | ~8 | 14× |
| **ALU utilization** | ~1% | ~75% | 75× |
| **Memory transactions** | 24 separate | 3 coalesced | 8× |
| **Instruction fetches** | 26 to ext mem | 2 to ext mem | 13× |
| **Data cache hits** | 0 | 12+ (L1/L2/SLC) | ∞ |
| **Program memory contentions** | 6 (Core 1 blocked) | 0 | N/A |

---

## Chapter 50.5: The Five Problems — Detailed Summary with Visualizations

Every problem visible in tiny-gpu maps to a specific Snapdragon hardware solution. Here's the complete breakdown:

### Problem 1: Instruction Fetch Bottleneck

**In tiny-gpu:** Every instruction fetch goes through the 1-channel program memory controller. Two cores share this channel. Core 1 always waits.

**Snapdragon solution:** Per-CU L0 instruction cache. After the first fetch, all accesses hit L0 in 1 cycle.

**Impact:** 13× fewer external memory fetches.

### Problem 2: No Memory Coalescing

**In tiny-gpu:** Each thread sends independent memory requests. 8 threads = 8 separate transactions.

**Snapdragon solution:** Coalescing unit merges consecutive addresses into 1 transaction (e.g., addresses 0-7 → 1 cache-line fetch).

**Impact:** 8× fewer memory transactions for stride-1 access.

### Problem 3: No Data Cache

**In tiny-gpu:** Every load goes to external memory. Second load costs same as first.

**Snapdragon solution:** Multi-level cache hierarchy (L1 → L2 → SLC → DRAM). Second load hits L1 cache.

**Visual comparison:**

```
tiny-gpu:        Load A[0] (6-8 cy) ... Load B[0] (6-8 cy) — same cost, no cache
Adreno 830:      Load A[0] (20 cy, DRAM miss) ... Load B[0] (4 cy, L1 hit) — 5× speedup
```

**Impact:** 5-25× faster for repeated/nearby accesses.

### Problem 4: Core Stalls on Memory

**In tiny-gpu:** Entire core stalls when ANY thread waits for memory. All 4 ALUs idle.

**Snapdragon solution:** Wave scheduling. When Wave 0 waits on memory, scheduler switches to Wave 1, 2, 3... ALUs never idle.

**Visual comparison:**

```
tiny-gpu:        8 ALUs × 113 cycles = 904 ALU-cycles. Only 8 used. Utilization: 0.9%
Adreno 830:      256 fibers × 60-95% utilization across all waves. Utilization: 60-95%
```

**Impact:** 60-95× better ALU utilization for real workloads.

### Problem 5: Unfair Memory Arbitration

**In tiny-gpu:** Fixed-priority arbiter always serves Core 0 first. Core 1 starves for 3+ cycles per memory instruction.

**Snapdragon solution:** QoS-aware arbitration with credit-based flow control. Round-robin or age-based scheduling prevents starvation.

**Impact:** Fair bandwidth allocation across all CUs.

---

## Chapter 50.6: What This Means for Real Inference

When you run an LLM on the Snapdragon 8 Elite at 70+ tokens/second, every optimization matters:

**Example: 7B parameter model at INT4**

```
Model: 7 billion parameters × 0.5 bytes = 3.5 GB weights
Snapdragon theoretical bandwidth: 76.8 GB/s (Qualcomm spec: ~84.8 GB/s)
NPU-available after SoC contention: ~65 GB/s

Conservative ceiling: 65 / 3.5 ≈ 19 tok/sec (no weight reuse)
Theoretical max:      76.8 / 3.5 ≈ 22 tok/sec

Qualcomm claims 70+ tok/sec. How?

Answer: Multiple simultaneous optimizations + favorable conditions

1. COALESCING: 8 requests → 1 transaction (reduces overhead)
   Effect: efficiency improvement but doesn't break bandwidth ceiling

2. CACHING: 44 MB on-chip SRAM holds ~88M params at INT4 (not 1B+)
   [44 MB ÷ 0.5 bytes/param = 88M params = 1.3% of 7B model]
   Effective for: KV cache, hot activations, embedding rows
   NOT effective for: bulk weight streaming (3.5 GB >> 44 MB)
   Effect: modest improvement for activation traffic

3. WAVE SCHEDULING: 40+ waves per CU = 6,400 threads in-flight
   Latency hidden behind parallel execution
   Effect: +20% throughput = 72 tok/sec

4. PRECISION ACCELERATION: INT4 multiply-accumulate = 1 cycle
   vs 3-4 cycles for FP32
   Effect: +10% throughput = 79 tok/sec

5. FUSION + TILING: Combine operations, reduce intermediate traffic
   Effect: already baked into above

Result: ~79 tok/sec ≈ "70+ tokens/second" claim
```

**Why tiny-gpu is the perfect teaching tool:**

tiny-gpu exposes all five problems explicitly. Once you understand these problems, the Snapdragon's solutions become obvious and justified. No feature is accidental — each solves a specific bottleneck visible in the educational GPU.

---

## Chapter 50.7: Phase 10 — What This Means for Real Inference (Corrected)

When you run an LLM on the Snapdragon 8 Elite, every problem we observed in tiny-gpu becomes a life-or-death performance issue at scale. But let me be precise about what we know, what we're estimating, and where the marketing ends and the physics begins.

### The Bandwidth Wall — Getting the Math Right

Let's start with what we **actually know:**

```
Snapdragon 8 Elite LPDDR5X:
  4 channels × 16-bit × 9600 MT/s / 8 bits per byte = 76.8 GB/s (theoretical peak)
  Qualcomm published spec: ~84.8 GB/s (higher-binned die or boosted MT/s bin)

  IMPORTANT: This 76.8–84.8 GB/s is the SoC-level total, shared by:
    CPU (Oryon cores), GPU (Adreno 830), NPU (Hexagon), ISP, display, modem
  NPU-available bandwidth in practice: ~50–70 GB/s (after OS/display/ISP overhead)

  This is the physical ceiling for the SoC. No software changes this number.
  Every byte of weight data that isn't cached must come through this shared pipe.
```

Now let's think about what LLM inference requires. A 7B parameter model quantized to INT4:

```
7 billion parameters × 4 bits / 8 bits per byte = 3.5 GB of weights
```

Here's the critical question: **how much of that 3.5 GB do you read per token?**

During autoregressive decode (generating one token at a time), each token requires a **matrix-vector multiply** through every layer. In the simplest case — dense attention, no sparsity, no MoE — you need to read **essentially all the weights** for every token. The activations are tiny, but the weight matrices are enormous.

```
Naive case (no caching, no reuse):
  Weights read per token: ~3.5 GB (all of them)
  At 70 tokens/second: 3.5 GB × 70 = 245 GB/s required

  LPDDR5X peak bandwidth: ~84.8 GB/s

  245 GB/s >> 84.8 GB/s

  THIS DOESN'T WORK.
```

So how does Qualcomm claim 70+ tokens/second? Several possibilities, likely in combination:

**Possibility 1: Smaller models.** The 70+ tok/s figure may be for models significantly smaller than 7B — perhaps 1-3B parameters.

```
3B params × INT4 = 1.5 GB weights
1.5 GB × 70 tok/s = 105 GB/s
Still above 84.8 GB/s peak, but achievable with compression + caching
```

**Possibility 2: On-chip weight caching.** The Snapdragon 8 Elite has significant on-chip memory:

```
CPU L2:     24 MB
GPU L2:     12 MB
SLC:         8 MB
NPU shared:  ? MB (not publicly specified)
─────────────────
Known total: 44+ MB of on-chip SRAM
```

If NPU caches can hold a meaningful fraction of model weights, you reduce DRAM reads. Layer-by-layer streaming is the realistic approach — cache one layer's weights, compute, evict, load next layer.

**Possibility 3: Operator fusion / Micro Tile Inferencing.** Qualcomm describes reducing **intermediate activation traffic** to DRAM by fusing multiple layers and keeping intermediate results in on-chip memory. Activations flowing between pipeline stages without DRAM round-trips.

```
Without fusion (tiny-gpu style):
  Layer 1: read weights from DRAM → compute → write activations to DRAM
  Layer 2: read weights from DRAM → read activations from DRAM → write to DRAM
  = 2 DRAM round-trips per layer for activations

With fusion:
  Load weight tiles for layers 1-N → keep activations on-chip →
  write final result to DRAM
  = Activation traffic to DRAM dramatically reduced
```

This doesn't reduce **weight** bandwidth (still streaming all weights), but eliminates large **activation** bandwidth, freeing DRAM bandwidth for weight streaming.

**Possibility 4: The number is benchmark-specific.** Reported tok/s varies enormously depending on:
- Model size and architecture
- Quantization scheme (INT4, INT8, mixed)
- Context length (longer context = more KV cache reads)
- Thermal state (sustained vs burst)
- Framework and runtime
- Prefill vs decode phase

**The honest answer:** We don't know exactly how Qualcomm achieves their claimed numbers because the NPU microarchitecture is proprietary. The bandwidth wall is real. Any solution must involve some combination of the above techniques.

### Connecting Back to tiny-gpu

The bandwidth wall is the scaled-up version of exactly what we see in `controller.sv`:

```sv
// From controller.sv — 8 consumers, 4 channels
for (int j = 0; j < NUM_CONSUMERS; j = j + 1) begin
    if (consumer_read_valid[j] && !channel_serving_consumer[j]) begin
        // This IS the bandwidth wall at tiny scale
        break;
    end
end
```

8 LSUs want data. 4 channels available. Some consumers wait. Same fundamental problem as "3.5 GB of weights need to flow through 84.8 GB/s of bandwidth."

| tiny-gpu Problem | Real Solution | Effect |
|---|---|---|
| Every thread sends separate request | **Coalescing** | Fewer transactions, less controller overhead |
| Every access goes to external memory | **Caching** — L1/L2/SLC | Repeated accesses from on-chip SRAM |
| Core stalls while waiting | **Latency hiding** — wave scheduling | ALUs stay busy |
| No data reuse | **Operator fusion** — on-chip intermediates | Reduce total DRAM traffic |
| Fixed-priority arbiter | **QoS-aware arbitration** | Fair scheduling |

### Problem 1 Revisited: Instruction Fetch Bottleneck

**In LLM inference:** Core compute loops are tight — same small instruction set executes billions of times. If every fetch went to DRAM like tiny-gpu, instruction delivery would dominate.

**Real solution:** Multi-level instruction caches. After the first execution of a loop, subsequent iterations are served from on-chip instruction cache. Exact cache sizes for Adreno 830 aren't publicly documented, but universal across modern GPUs.

### Problem 2 Revisited: No Coalescing

**In LLM inference:** Loading weight matrices means many threads loading elements. Without coalescing, each element is a separate transaction with its own address decode and row activation overhead.

**Real solution:** Hardware coalescing units merge requests to adjacent addresses. For stride-1 access (natural for matrix operations), reduces transaction count by coalescing window width (typically 32-128 bytes).

### Problem 3 Revisited: No Cache

**In LLM inference:** Attention mechanism accesses KV cache repeatedly. Without caching, every access requires streaming from DRAM.

**Real solution:** Multi-level cache hierarchy. Snapdragon 8 Elite has 44+ MB known on-chip SRAM, plus unknown NPU-internal SRAM. SLC shared across all IP blocks acts as bandwidth amplifier.

### Problem 4 Revisited: Core Stalls on Memory

**In LLM inference:** Matrix-vector multiplies interleave weight loads with computation. Stalling on every load drops utilization to single digits.

**Real solution - Adreno 830 GPU:** Wave scheduling. Multiple waves resident per CU, scheduler switches between them on stalls, ALUs never idle.

**Real solution - Hexagon NPU:** Tensor accelerator uses systolic array / MAC array where data flows through predetermined pattern. Memory system designed to feed data at array's consumption rate — **dataflow architecture** rather than load-stall-compute.

```
Systolic array:
  Data flows through continuously.
  Memory prefetch timed to match consumption rate.
  Near 100% MAC utilization when bandwidth permits.
```

### Problem 5 Revisited: Unfair Arbitration

**On the Snapdragon SoC:** Multiple IP blocks compete for 4 LPDDR5X channels:

```
NPU:       Streaming weights (latency-sensitive)
GPU:       Rendering UI (moderate priority)
ISP:       Processing camera (real-time deadline)
Display:   120 Hz refresh (hard deadline)
CPU:       Inference framework (moderate)
Modem:     Network traffic (bursty)
```

If using tiny-gpu's fixed-priority scheme, display or ISP would starve NPU. NoC fabric uses QoS-aware arbitration — deadline-driven clients guaranteed bandwidth, compute clients share remainder fairly.

---

## Chapter 50.8: Phase 11 — The Power Dimension

### Why tiny-gpu's Architecture Would Burn Through Battery

Let's count the wasted energy in tiny-gpu:

**Wasted instruction fetches:**

```
13 instructions × 2 cores × ~4 cycles/fetch = ~104 fetch cycles

Energy per external memory access: ~10-100× more than cache hit
With I-cache: ~13 cycles, ~10-100× less energy per access
```

**Wasted ALU idle cycles:**

```
~104 total cycles, ~8 useful ALU cycles = 96 idle cycles

Idle ALUs still consume leakage power
Clock tree still toggles, consuming dynamic power

With wave scheduling: ALUs busy ~95% of cycles
Same computation, ~20× less energy wasted on idle hardware
```

**Wasted memory bandwidth:**

```
8 separate memory transactions where 1 coalesced transaction would suffice

8× more DRAM activations = 8× more DRAM energy
With coalescing: 1 transaction, 1 row activation, 8× less DRAM energy
```

### How the Snapdragon Manages Power

The PM8750 PMIC provides **independent power rails** for different IP blocks:

```
PM8750 PMIC
├── VddCx    → CPU digital logic
├── VddGfx   → GPU (Adreno 830) — per-slice power gating
├── VddNPU   → NPU tensor accelerator — INDEPENDENT DVFS
├── VddMx    → Memory logic
├── VddMss   → Modem subsystem
└── VddCam   → ISP / camera
```

The NPU tensor accelerator having its **own power rail** is critical for inference:

```
During LLM inference:
  Tensor accelerator: Running at max frequency/voltage (doing matmuls)
  Scalar cores: Running at low frequency (just control flow)
  Vector cores: Running at medium frequency (activations, softmax)
  GPU: Power-gated (not needed for inference)
  CPU Prime cores: Power-gated (inference running on NPU)
  CPU Perf cores: 1 core at low frequency (running inference framework)
```

This means the NPU can run at peak performance while other components are in low-power modes. tiny-gpu has no concept of power management — both cores always run at full power.

---

## Chapter 50.9: Phase 12 — The Complete Picture

### Scale Comparison

```
                        tiny-gpu matadd          Snapdragon LLM inference
                        ───────────────          ────────────────────────
Computation:            8 additions              ~7 billion MACs per token
Data size:              16 bytes input           ~3.5 GB weights + KV cache
Threads:                8                        Millions (across 32 layers)
Memory bandwidth used:  ~32 bits/cycle           ~40 GB/s sustained
Compute throughput:     ~0.08 ops/cycle          ~75 TOPS (INT8)
Power consumption:      N/A (simulation)         ~4-5W sustained
Result:                 8 sums                   1 token (70+ per second)
Time:                   ~104 cycles              ~14 ms per token
```

The **architecture is identical.** Every tiny-gpu concept maps directly:

```
tiny-gpu concept           →  What it becomes at scale
────────────────           →  ──────────────────────────
dcr.sv (1 register)        →  Command Processor (thousands of registers)
dispatch.sv (2 cores)      →  Workgroup scheduler (12 CUs / 6 SPs)
core.sv (4 threads)        →  Compute Unit (thousands of fibers)
scheduler.sv (sequential)  →  Wave scheduler / systolic dataflow
fetcher.sv (no cache)      →  Instruction fetch + L0/L1 I-cache
decoder.sv (11 opcodes)    →  Decode unit (thousands of opcodes)
alu.sv (8-bit ops)         →  FP32/FP16/INT8/INT4 ALUs + tensor cores
lsu.sv (1 request at time) →  LSU + coalescing + cache hierarchy
registers.sv (16×8-bit)    →  256KB+ register file per CU
controller.sv (4 channels) →  NoC fabric + SLC + 4-channel LPDDR5X
memory.py (256 bytes)      →  12 GB LPDDR5X, banked, cached, compressed
```

### The Software Path for LLM Inference

```
Samsung Galaxy AI app
  │  "Summarize this article"
  ▼
Android NNAPI / QNN SDK
  │  Model: Llama 2 7B INT4 (~3.5 GB)
  │  Compiled: matmul → layernorm → attention → ...
  ▼
QNN Driver (fastrpc IPC to CDSP)
  │  Sends compiled graph to Hexagon firmware
  │  Allocates weight buffers in LPDDR5X
  │  Maps buffers into NPU's address space via SMMU
  ▼
Hexagon NPU Firmware
  │  Micro tile scheduler:
  │    "Layers 1-4 weights fit in shared memory"
  │    "Execute without DRAM round-trip"
  ▼
Tensor Accelerator Hardware
  │  Systolic array: weights × activations → accumulate
  │  INT4 weights unpacked to INT8 for computation
  ▼
Vector Cores (HVX)
  │  ReLU / GELU / SiLU activations
  │  Softmax for attention
  │  LayerNorm
  ▼
Scalar Cores
  │  Control flow: loop counters, tile indexing
  │  Token sampling from probability distribution
  ▼
Result: next token → app via fastrpc
  │
  │  Repeat 70+ times per second
  ▼
"Here is a summary of the article..."
```

Every step has a direct ancestor in tiny-gpu:

- **Tensor accelerator** doing matmul = tiny-gpu's `alu.sv` scaled to billions of ops
- **Micro tile scheduler** = tiny-gpu's `scheduler.sv` evolved to avoid memory stalls
- **Shared memory** = what `controller.sv` would become with caching
- **fastrpc IPC** = sophisticated version of tiny-gpu's `setup.py`
- **SMMU** (IOMMU) = needed when tiny-gpu's `memory.py` must share address space with other IP blocks

### The Final Insight

tiny-gpu is ~1000 lines of SystemVerilog showing **every fundamental GPU architecture problem:**

1. **Instruction delivery** — `fetcher.sv` shows why you need I-cache
2. **Memory bandwidth** — `controller.sv` shows why you need coalescing
3. **Memory latency** — `lsu.sv` + `scheduler.sv` show why you need caching and wave scheduling
4. **Compute utilization** — WAIT state shows why you need latency hiding
5. **Fairness** — priority arbiter shows why you need QoS-aware scheduling
6. **SIMD execution** — `registers.sv` with `THREAD_ID` shows how threads run same code on different data
7. **Work distribution** — `dispatch.sv` shows how blocks get assigned to cores

The Snapdragon 8 Elite's Adreno 830 and Hexagon NPU solve every one of these problems with billions of transistors on a 3nm process. But the **structure** — dispatcher → cores → scheduler → ALU/LSU/registers → memory controller → external memory — is identical.

Understanding tiny-gpu means understanding the **skeleton that every GPU is built on**, from Adreno 830 to NVIDIA B200. The optimizations (caching, coalescing, wave scheduling, divergence handling, power management) are the meat on the bones. But the bones are right here in these 12 SystemVerilog files.

---

# Part VIII: The Bandwidth Wall and Real Inference

## Chapter 51: The Bandwidth Wall — Applied to Real Hardware

### The Bandwidth Ceiling: Physics-Based

```
tokens/sec ≤ DRAM_bandwidth ÷ bytes_of_weights_per_token

┌─────────────────────────── Snapdragon 8 Elite ─────────────────────────────┐
│                                                                             │
│  LPDDR5X: 4ch × 16-bit × 9600 MT/s = 76.8 GB/s theoretical peak           │
│           Qualcomm spec: ~84.8 GB/s; NPU share: ~50–70 GB/s in practice    │
│                                                                             │
│  Model     Precision   Weight bytes   Ceiling (tok/s)   [full BW; no reuse]│
│  ────────  ─────────   ────────────   ───────────────                      │
│  7B param  INT4        3.5  GB        76.8 / 3.5  ≈  22  (theoretical max) │
│  3B param  INT4        1.5  GB        76.8 / 1.5  ≈  51                   │
│  1B param  INT4        0.5  GB        76.8 / 0.5  ≈ 154                   │
│                                                                             │
│  "70+ tok/s" requires: small model (≤3B), weight reuse, or prefill burst   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── NVIDIA B200 ────────────────────────────────────┐
│                                                                             │
│  HBM3e: ~8 TB/s per GPU                                                    │
│                                                                             │
│  70B param  FP4   35 GB   →   8,000 / 35  ≈ 228 tok/s per GPU             │
│                              + 8 GPUs w/ model parallelism = 1000s tok/s  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bandwidth Comparison (Log Scale)

```
Snapdragon  │███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  ~0.08 TB/s
B200        │████████████████████████████████████████████████████████████████████████████████████████████████████│  ~8.0 TB/s
            0                                                            8.0 TB/s  (100× difference)
```

### Snapdragon 8 Elite (LPDDR5X)

For a 7B parameter model at INT4:

```
Weight payload: 7 × 10⁹ × 0.5 bytes = 3.5 GB per token (bytes loaded, not bytes/sec)

If streaming all weights from DRAM per token (no reuse, full SoC bandwidth):
  tokens/sec ≤ 76.8 GB/s ÷ 3.5 GB/token ≈ 22 tok/s (theoretical ceiling)

Accounting for SoC contention (~65 GB/s NPU-available):
  tokens/sec ≤ 65 / 3.5 ≈ 19 tok/s (more realistic ceiling)

Accounting for weight reuse (reuse_factor r, 0 ≤ r < 1):
  tokens/sec ≤ BW / (3.5 × (1 - r))
  At r=0.10: ≤ 65 / 3.15 ≈ 21 tok/s
  At r=0.50: ≤ 65 / 1.75 ≈ 37 tok/s  [requires significant on-chip SRAM hits]

This is a physics ceiling for the no-reuse case. On-chip caching can raise it —
but only for the fraction of weights that physically fit in SRAM (see §53.5).
```

When Qualcomm claims "70+ tokens/second," the bandwidth math tells us one or more of the following must be true:

1. **The model is smaller than 7B.** A 1-3B model makes the number physically plausible.
2. **On-chip caching provides some weight reuse.** 44+ MB known on-chip SRAM.
3. **The number is for prefill, not decode.** Prefill has higher parallelism.
4. **The number is burst, not sustained.** Thermal throttling degrades over time.
5. **Compression or sparsity reduces effective weight reads.**

### NVIDIA B200 (HBM3e)

```
Single B200 GPU: ~8 TB/s HBM3e bandwidth
DGX B200 (8 GPUs): 1,440 GB total HBM3e, 64 TB/s aggregate

For 70B model at FP4:
  Weight payload: 70 × 10⁹ × 0.5 = 35 GB
  tokens/sec ≤ 8,000 / 35 ≈ 228 tokens/sec per GPU
```

### The Arithmetic Intensity Gap: Why Chips Can't Breathe

"How many bytes does the tensor accelerator need to stay fully busy?"

**Snapdragon Hexagon NPU:**
```
┌─────────────────────────────────────────────────────────────┐
│  Peak compute:              75 TOPS INT8                    │
│                                                             │
│  Required feed = compute / arithmetic_intensity (AI):       │
│    required_BW = 75 TOPS ÷ AI                               │
│                                                             │
│  Decode  (batch=1, AI ≈ 1–4 FLOPS/byte):                  │
│    75 TOPS ÷ 2 FLOPS/byte ≈ 37 TB/s >> DRAM (0.085 TB/s)  │
│    Gap: 37 / 0.085 ≈ 435×  → severely bandwidth-bound      │
│                                                             │
│  Prefill (tiled GEMM, AI ≈ 100–500 FLOPS/byte):            │
│    75 TOPS ÷ 300 FLOPS/byte ≈ 0.25 TB/s > DRAM            │
│    Gap: 0.25 / 0.085 ≈ 3×  → approaching compute-limited   │
│                                                             │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ DRAM
│  ██████████████████████████ (×435 in batch=1 decode)│ Needed│
└─────────────────────────────────────────────────────────────┘
→ Arithmetic intensity (AI) determines the gap, not a fixed multiplier.
→ On-chip tiling raises AI — that's exactly why tiling is not optional.

NVIDIA B200:
┌─────────────────────────────────────────────────────────────┐
│  Peak compute:              ~2,500 TOPS FP8                 │
│                                                             │
│  Required feed = 2,500 TOPS ÷ AI:                          │
│    Decode  (AI ≈ 1–4):  625–2,500 TB/s  >> HBM (8 TB/s)   │
│    Prefill (AI ≈ 1000): 2.5 TB/s        <  HBM (8 TB/s) ✓ │
│                                                             │
│  Decode gap:    2,500 TB/s needed / 8 TB/s HBM = 312×      │
│  Prefill gap:   2.5 TB/s needed / 8 TB/s HBM < 1× ← WIN   │
│                                                             │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│ HBM
│  ████████████████████████████████████ (×312 in decode)│ Need│
└─────────────────────────────────────────────────────────────┘
HBM's 100× bandwidth advantage means B200 can approach compute-limited
behavior in large-batch prefill when AI ≥ ~312 FLOPS/byte.

In practice, real transformer kernels are partially compute-bound:
  - Layernorm, softmax, elementwise ops have much lower AI
  - KV cache writes add bandwidth pressure during prefill
  - Tiling imperfections leave some compute cycles idle
  - Memory pipeline overlap is never 100%
→ B200 prefill is "significantly less bandwidth-bound than decode"
  but rarely hits pure compute saturation on full transformer graphs.
Both chips require on-chip reuse; HBM narrows the gap dramatically.
```

### Hardware Face-Off: Same Physics, Different Budgets

```
                       ┌──────────────────────┬──────────────────────┐
                       │  Snapdragon 8 Elite  │   NVIDIA B200 (1 GPU)│
─────────────────────────┼──────────────────────┼──────────────────────┤
DRAM Bandwidth       │  ~0.08 TB/s          │  ~8 TB/s             │
                       │  ░░░                 │  ████████████████████│
─────────────────────────┼──────────────────────┼──────────────────────┤
Power Budget         │  ~5W sustained       │  ~1,000W TDP         │
                       │  ░                   │  ████████████████████│
─────────────────────────┼──────────────────────┼──────────────────────┤
On-package Memory    │  12 GB LPDDR5X       │  192 GB HBM3e        │
                       │  ██                  │  ████████████████████│
─────────────────────────┼──────────────────────┼──────────────────────┤
Transistors          │  ~20B                │  ~208B               │
                       │  ██                  │  ████████████████████│
─────────────────────────┼──────────────────────┼──────────────────────┤
Process Node         │  3nm (TSMC N3E)      │  4nm (TSMC N4)       │
                       │  similar ─────────────────────── similar   │
─────────────────────────┴──────────────────────┴──────────────────────┘
Both platforms: more compute than memory bandwidth can feed.
Goal is identical: maximize reuse per byte fetched from DRAM.
```

---

## Chapter 52: LLM Inference Mapped to tiny-gpu Problems

### Instruction Fetch → Solved by I-Cache and Fixed-Function Datapaths

The transformer computation is a tight loop — same instructions billions of times. Modern GPUs have instruction cache hierarchies. NPU tensor accelerators use fixed-function datapaths with tightly controlled microcode, dramatically reducing instruction overhead compared to a general-purpose core.

### No Coalescing → Solved by Hardware Coalescing

Loading weight matrix rows means thousands of threads loading one element each. Without coalescing: millions of transactions. With coalescing: thousands of wider transactions.

### No Cache → Solved by Multi-Level Cache + Micro Tile Inferencing

Qualcomm describes "Micro Tile Inferencing" — reducing intermediate activation traffic to DRAM by fusing multiple layers and keeping intermediate results in on-chip memory. From Qualcomm's descriptions:

> "Microtile inferencing breaks networks into microtiles executed independently and can eliminate memory traffic between as many as 10 or more layers."

This primarily targets **activation bandwidth**, not the fundamental need to consume weights. It frees external bandwidth for the weight stream.

The mechanism for throughput improvement is **intra-tile arithmetic intensity**, not percentage of model cached:

```
Standard layer-by-layer (no fusion):
  For each layer: load weights → compute → write activations to DRAM → load next
  Activation DRAM traffic: O(seq_len × hidden_dim) per layer boundary
  Weight AI: ~2 FLOPS/byte (still bandwidth-bound for weights)

Micro-tiled fusion (fuse 10+ layers):
  Load weights for fused block → keep activations in SRAM → compute all layers → write only final output
  Activation DRAM traffic: reduced by ~10× for fused region
  Weight AI: still ~2 FLOPS/byte per weight byte (weights must still be read)
  Net effect: DRAM bandwidth freed for weight stream + reduced total traffic

Key insight: Micro-tiling raises effective compute-per-DRAM-byte by eliminating
activation round-trips. It does NOT reduce weight bandwidth; it uses freed BW
capacity to accommodate the weight stream more efficiently.
```

### Core Stalls → Solved by Wave Scheduling / Systolic Dataflow

The Hexagon NPU's tensor accelerator uses a **systolic array** where data flows through MAC units in a predetermined pattern:

```
Weight[0] → MAC → MAC → MAC → MAC → accumulate
Weight[1] → MAC → MAC → MAC → MAC → accumulate
             ↑     ↑     ↑     ↑
           Act[0] Act[1] Act[2] Act[3]
```

Memory prefetch timed to match consumption rate. Near 100% MAC utilization when bandwidth keeps up.

### Unfair Arbitration → Solved by QoS-Aware NoC

Multiple IP blocks compete for the same 4 LPDDR5X channels. Display (120 Hz deadlines) gets guaranteed bandwidth. ISP (real-time camera) gets guaranteed bandwidth. NPU and GPU share the remainder with fairness.

---

## Chapter 53: What We Know vs What We Don't

```
WHAT WE KNOW (from physics + public specs):
  - LPDDR5X theoretical bandwidth: 76.8 GB/s (4ch × 16-bit × 9600 MT/s ÷ 8)
    Qualcomm spec: ~84.8 GB/s; NPU-available after SoC contention: ~50–70 GB/s
  - 7B INT4 model weights: ~3.5 GB
  - Streaming all weights per token at 70 tok/s would require ~245 GB/s
  - 245 > 84.8 → something else is going on
  - On-chip SRAM (known): 44+ MB across CPU/GPU/SLC
  - NPU rated at ~75 TOPS INT8

WHAT WE KNOW from Qualcomm's descriptions:
  - Fused scalar/vector/tensor architecture with shared on-chip memory
  - "Micro Tile Inferencing" reduces intermediate DRAM traffic
  - Independent power rail for tensor accelerator
  - Native INT4 support
  - 70+ tok/s claimed (model/conditions unspecified)

WHAT WE DON'T KNOW (proprietary):
  - Exact NPU internal SRAM size
  - Exact cache hierarchy within Adreno 830 CUs
  - Whether SLC supports per-client partitioning
  - Exact model/quantization for the 70 tok/s claim
  - Sustained vs burst under thermal constraints
  - Specific coalescing window sizes
```

---

## Chapter 53.5: Prefill vs Decode — Two Radically Different Workloads

The "70+ tok/s" claim for Snapdragon masks a critical distinction: **prefill and decode are fundamentally different problems with different bandwidth requirements.**

### Prefill: Batch Parallelism

When you start inference (processing the initial prompt), all tokens in that prompt are independent. This is called "prefill."

```
Prompt: "The quick brown fox jumps"

Tokens:            [The]  [quick] [brown] [fox]  [jumps]
                     ↓      ↓       ↓       ↓      ↓
Work per token:    1 pass 1 pass  1 pass  1 pass 1 pass    (all in parallel)

Weight load for 6-token prefill on 7B model:
  - Need: 7B weights
  - Tokens: 6
  - Parallelism: 6× (all tokens in one batch)

Operational Intensity (FLOP/byte):
  matmul([batch=6, seq_len=1, dim=4096], [dim=4096, d_ff=11008])
  = 2 × 6 × 1 × 4096 × 11008 FLOPS ÷ (6 × 4096 × 4 bytes)
  ≈ 2 × 11008 / 4 ≈ 5,500 FLOPS/byte  ← compute-limited at this batch size
  [NOTE: This 5,500 FLOPS/byte assumes batch=6 and ignores KV cache bandwidth.
   Actual AI depends on seq_len, batch_size, hidden_dim, and SRAM reuse depth.
   At batch=1, AI ≈ 1–4 FLOPS/byte (bandwidth-bound, like decode).]
```

**Prefill can run at peak compute efficiency** because you're amortizing weights across multiple tokens. This is essentially a tiled matrix-matrix multiply (GEMM).

### Decode: Token Generation

After prefill, LLMs generate one token at a time. This is "decode" or "auto-regressive generation."

```
Previous output: [The, quick, brown]

Current step:
  - Input: [The, quick, brown] (3 tokens)
  - Compute: 7B weights × 3-token batch
  - Output: 1 new token

Next step:
  - Input: [The, quick, brown, fox] (4 tokens)
  - Compute: 7B weights × 4-token batch
  - Output: 1 new token
```

But **for each new token, you still load the full 7B weight matrix once:**

```
Weight load per new token:
  - Input: N tokens
  - Weights loaded: 7B (entire model)
  - Computation: O(N × hidden_dim × weights)

Operations ÷ Bytes:
  matmul([batch=N, seq_len=1, dim=4096], [dim=4096, d_ff=11008])
  = 2 × N × 4096 × 11008 / (7B × 4 bytes for weights)
  = 2 × N × 4096 × 11008 / (28 × 10⁹ × 4)
  ≈ N × 0.008 FLOPS/byte ← BANDWIDTH-LIMITED!
```

**Decode operational intensity is proportional to batch size.** At batch=1 (online inference), it's ~1-4 FLOPS/byte.

### The Bandwidth Ceiling Equation (Derived from First Principles)

For decode, the ceiling is **hard physics**:

```
tokens_per_second = BW_available / (model_size_bytes × (1 - reuse_factor))

Where:
  BW_available   = DRAM_BW × NPU_allocation_fraction
                 = 76.8 GB/s × ~0.85  ≈  65 GB/s (after SoC contention)
  model_size     = 3.5 GB  (7B INT4, bytes per token — not bytes/sec!)
  reuse_factor   = fraction of weights already in on-chip SRAM (0 ≤ r < 0.013 max)

Zero-reuse ceiling (conservative):
  tokens/sec ≤ 65 / 3.5 ≈ 19 tok/s

With 1% SRAM cache hit:
  tokens/sec ≤ 65 / (3.5 × 0.99) ≈ 19 tok/s  [negligible improvement]

The ceiling is real. The path to higher throughput is smaller models, not caching 7B weights.
```

### Why "70+ tok/s" Is Theoretically Possible (But Needs Conditions)

Four mechanisms could exceed 24 tok/s:

**1. Smaller model**
```
1B model, INT4:
  tokens/sec ≤ 76.8 / 0.5 ≈ 154 tok/s (theoretical ceiling)
  NPU-available BW: ≤ 65 / 0.5 ≈ 130 tok/s (realistic ceiling)
This is physically plausible. But it's a 7× smaller model with lower capability.
```

**2. Caching (on-chip SRAM reuse)**
```
If you keep some weights in 44 MB of on-chip SRAM:
  - First token: load from DRAM (slow)
  - Subsequent tokens: hit SRAM cache (fast)

Reality check on caching:
  On-chip SRAM (all Snapdragon caches combined): ~44 MB
  7B INT4 model size: 3,500 MB
  Maximum cacheable fraction: 44 / 3,500 ≈ 1.3%

Correct time-per-token analysis (DRAM remains the bottleneck):
  Per token, you need 3.5 GB of weight data:
    SRAM-cached (1.3%):  45.5 MB  at 10,000 GB/s → 0.005 ms  (negligible)
    DRAM-loaded (98.7%): 3,454 MB at 76.8 GB/s   → 45.0 ms   (bottleneck!)

  Time per token: 45.0 ms vs 45.6 ms (no cache) → tokens/sec ≈ 22.2 tok/s
  Improvement from 1.3% weight caching: ~1%  ← essentially zero

Why? SRAM is so fast (10 TB/s) that even 1.3% of accesses happen in
0.005 ms, while DRAM takes 45 ms. You can't hide 45 ms behind 0.005 ms.
The DRAM is still the bottleneck; it still saturates the 76.8 GB/s channel.

What IS cacheable (and helpful for different reasons):
  - Attention KV cache: reduces attention computation FLOP count, not weight BW
  - Hot activation layers: fused microtile kernels keep activations in SRAM,
    freeing DRAM for the weight stream (net positive for bandwidth budget)
  - Frequently-accessed embeddings for common tokens (tiny, fits in cache)
  - NOT: the bulk of weight matrices (3.5 GB >> 44 MB SRAM capacity)
```

**3. Prefill burst**
```
During prefill, operational intensity is 1000+ FLOPS/byte.
You CAN hit 70+ tok/s during prefill. But prefill is measured in seconds or tens of seconds (preparing the prompt). It's not sustained inference.
```

**4. Quantization savings**
```
Extreme quantization (binary or ternary weights) could reduce effective model size below 0.5 GB. But this requires acceptance of severe accuracy loss.
```

### What the Physics Actually Says

```
┌─────────────────────────────────────────────────────────────────┐
│ SUSTAINED decode at 70+ tok/s on Snapdragon requires:           │
│                                                                 │
│ (A) Model size < 1.2 GB (≤1B parameters at INT4)               │
│  OR                                                             │
│ (B) Intra-tile weight reuse (higher AI within each SRAM tile)   │
│  OR                                                             │
│ (C) Both prefill and inference, not pure decode                │
│                                                                 │
│ Without one of these, the physics says: ~24 tok/s max.         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Chapter 53.6: Quantization Effects on Bandwidth — The Full Math

Quantization doesn't just save storage. It fundamentally changes the bandwidth requirement and operational intensity of every computation.

### How Quantization Changes Bandwidth Demand

```
For a 7B parameter LLM during decode:
[Using 76.8 GB/s theoretical ceiling; NPU-available ~65 GB/s after contention]

FP32 (32 bits per weight):
  Model size = 7B × 4 bytes = 28 GB
  tokens/sec ≤ 76.8 / 28 ≈ 3 tok/s  [NPU-avail: 65/28 ≈ 2.3 tok/s]

FP16 (16 bits per weight):
  Model size = 7B × 2 bytes = 14 GB
  tokens/sec ≤ 76.8 / 14 ≈ 5 tok/s  [NPU-avail: 65/14 ≈ 4.6 tok/s]

INT8 (8 bits per weight):
  Model size = 7B × 1 byte = 7 GB
  tokens/sec ≤ 76.8 / 7 ≈ 11 tok/s  [NPU-avail: 65/7 ≈ 9.3 tok/s]

INT4 (4 bits per weight):
  Model size = 7B × 0.5 bytes = 3.5 GB
  tokens/sec ≤ 76.8 / 3.5 ≈ 22 tok/s [NPU-avail: 65/3.5 ≈ 19 tok/s]

INT2 (2 bits per weight — extreme):
  Model size = 7B × 0.25 bytes = 1.75 GB
  tokens/sec ≤ 76.8 / 1.75 ≈ 44 tok/s (theoretical ceiling only)
  But: 2-bit quantization destroys model accuracy badly
```

### Quantization Quality vs Bandwidth Tradeoff

```
┌────────────────────────────────────────────────────────────────┐
│ Bandwidth (GB/s) needed per token for 7B model                 │
│                                                                │
│  40  ████████████████████  FP32  (pure float)                 │
│  20  ██████████  FP16  (half precision)                       │
│  10  █████  INT8  (mixed-precision quantization)              │
│   5  ██ INT4  (well-established post-training)                │
│   2  █ INT2  (research only; severe accuracy loss)            │
│   1  INT1  (binarized; basically unusable)                   │
│   0  ────────────────────────────────────────────────────────│
│       0   10   20   30   40                                   │
│       ↑                                                        │
│   Snapdragon 8 Elite: 76.8 GB/s theoretical / ~60–70 GB/s sustained │
│   At 22 tok/s: 22 × 3.5 GB/token = 77 GB/s ≈ at or beyond sustained│
│   (real LPDDR sustained BW ~60–70 GB/s after refresh + overhead)    │
└────────────────────────────────────────────────────────────────┘
```

### Why Snapdragon "Underutilizes" Bandwidth at High Quantization

```
Snapdragon LPDDR5X: ~76.8 GB/s theoretical (NPU-available: ~60–70 GB/s)
Required for 7B INT4: 3.5 GB/token (not GB/s!)

At 22 tok/s throughput:
  Bandwidth consumed = 22 tok/s × 3.5 GB/token = 77 GB/s
  vs. theoretical peak:  76.8 GB/s  → already at or beyond theoretical peak
  vs. sustained reality: ~60–70 GB/s (LPDDR5X loses ~10–15% to refresh
    cycles, command overhead, and rank interleave inefficiency under load)

  This means 22 tok/s requires near-ideal LPDDR streaming conditions:
  minimal page conflicts, sustained burst mode, no SoC contention spikes.
  Real-world sustained decode is more likely 16–20 tok/s for a 7B INT4 model.

At 10 tok/s (thermal throttled):
  Bandwidth consumed = 10 × 3.5 = 35 GB/s
  Utilization: 35 / 76.8 ≈ 46%  ← headroom for CPU/ISP traffic

This looks wasteful. But it's actually accurate:
  - Compute is cheap (FP32 multiply ≈ 3.7 pJ)
  - Data movement is expensive (DRAM read ≈ 640 pJ)
  - Even with extreme quantization, you're still bandwidth-bound

The extra bandwidth capacity is used for:
  - Prefill (which needs higher BW due to batching)
  - Prefetching (overlapping future weight loads)
  - Write-back of intermediate activations
  - CPU/GPU/NPU memory traffic sharing
```

### Cascading Quantization Effects: Activations, Weights, Gradients

During inference, quantization applies to:

```
FP32 Baseline (inference only, no gradients):
  Weights:     28 GB
  Activations: ~1 GB per token (intermediate results)
  Total load:  ~29 GB per token

INT4 Weights + FP16 Activations:
  Weights:     3.5 GB (quantized)
  Activations: 0.5 GB (kept precise for stability)
  Total:       ~4 GB per token ✓ Much better

INT4 Weights + INT8 Activations:
  Weights:     3.5 GB
  Activations: 0.25 GB
  Total:       ~3.75 GB per token (aggressive but risky)
```

### Why GPUs and NPUs Differ in Quantization Support

```
NVIDIA B200:
  - 8 TB/s HBM bandwidth
  - Can support FP32 + FP32 activations
  - Can run FP4 quantization losslessly on 70B model
  - tokens/sec = 8000 / 35 = 228 tok/s

Snapdragon Adreno 830:
  - 84.8 GB/s LPDDR5X bandwidth
  - Must quantize weights to INT4/INT8
  - Even then: 24 tok/s for 7B
  - Adreno tensor unit has dedicated INT4 hardware (no FP32 penalty)
  - Qualcomm chose INT4-native design because DRAM BW is the constraint
```

The difference reflects **architectural constraint awareness**: B200 has bandwidth to burn. Snapdragon's design explicitly optimizes for aggressive quantization.

---

## Chapter 53.7: Systolic Arrays vs Dataflow — Two Memory Hierarchies

The tensor accelerators in B200 and Snapdragon implement fundamentally different execution paradigms. This affects bandwidth, latency, and power.

### Systolic Arrays (Spatial Dataflow)

A systolic array is a grid of processing elements (PEs) where data flows in predetermined paths:

```
Simplified 4×4 Systolic Array (B200 Tensor Core style):

        Input A (rows)
           │  │  │  │
           ↓  ↓  ↓  ↓
        ┌──────────────┐
        │ PE  PE  PE  PE │
        │ PE  PE  PE  PE │ ← data flows right (A)
        │ PE  PE  PE  PE │
        │ PE  PE  PE  PE │
        └──────────────┘
           ↓  ↓  ↓  ↓
     Output C (results)

(B flows downward)

Key property: Once A and B are loaded, computation is **perfectly pipelined** with zero stalls for memory.

Example: 4×4 × 4×4 matrix multiply
  FMA operations:     4 × 4 × 4 = 64
  Cycles to compute:  ~4 (latency of longest path)
  Stalls:            0 (no memory waits; fully pipelined)
```

**Systolic pros:**
- Perfect pipeline utilization (no memory stalls)
- Deterministic latency
- Minimal register file (data flows through, doesn't stop)

**Systolic cons:**
- Fixed data flow pattern (not flexible)
- Requires large matrix to amortize loading cost
- Not suitable for irregular computation

### Dataflow (Register-and-Forward)

B200's approach is closer to **spatial + temporal dataflow** — data flows but is explicitly managed by register files:

```
B200-style Distributed Memory Architecture:

  Tensor Memory (TMEM) — new in Blackwell
         │
         ↓ (data read)
  ┌─────────────────────┐
  │ Register File (256KB)│ ← local per SM
  │      ↓              │
  │  Tensor Core Array  │ ← compute
  │      ↓              │
  │  (result written)   │
  └─────────────────────┘
         │
         ↓ (write-back)
  L2 Cache / HBM

Key difference from systolic:
  Data can be **reused** from registers across multiple operations.
  Memory hierarchically managed (TMEM vs HBM vs register).
```

### Why Snapdragon Uses a Hybrid Approach

The Snapdragon NPU is described as **"fused scalar/vector/tensor"**. This suggests:

```
Snapdragon Execution Model (inferred):

Shared On-Chip Memory (~44 MB SLC + CPU L1/L2):
         │
         ├─→ Tensor Accelerator (systolic-like)
         ├─→ Vector ALUs (scalar in Hexagon)
         └─→ Adreno GPU (dataflow-based)

Advantage: Each can prefetch different data independently.
  - Tensor: load matrix A while computing B
  - Vector: load scalar states
  - GPU: load graphics
  → Aggregate bandwidth utilization is higher

This is why Snapdragon can claim reasonable performance despite ~85 GB/s limit.
```

### Memory Hierarchy Implications

```
                NVIDIA B200           Snapdragon 8 Elite
Compute         Tensor cores          Tensor accel + GPU + Hex
Register File   256 KB per SM          Distributed (TMEM, shared)
L1 Cache        Per-SM, ~30 cy         Per-CU, ~4 cy
L2 Cache        Shared, ~200 cy        Slice cache, ~30-40 cy
SLC Cache       —                      8 MB shared, ~30-40 cy
DRAM            HBM3e, ~400 cy         LPDDR5X, ~100-150 cy

B200 design favors: Large working sets, temporal reuse
Snapdragon design favors: Spatial locality, QoS arbitration
```

---

## Chapter 53.8: Shared Memory and Micro Tiling — Keeping Data On-Chip

Both Snapdragon and B200 use **shared memory (scratchpad SRAM)** to break the DRAM barrier. The techniques differ:

### Snapdragon's "Micro Tile Inferencing"

Qualcomm explicitly mentions **"Micro Tile Inferencing"** — fused layers on small matrix blocks.

```
Standard inference (tiny-gpu style):
  Layer 1:  [compute A] → write A to DRAM
            [compute B] → write B to DRAM
  Layer 2:  [load A from DRAM] → [compute C]
            [load B from DRAM] → [compute D]

  Memory traffic: A and B written, then loaded = 2× bandwidth

Micro Tiling (on-chip fusion):
  Tile = 64×64 submatrix of hidden layer

  Step 1: Load tile of A from DRAM → on-chip SRAM (1 MB)
          Load tile of weights → on-chip SRAM (2 MB)
          Compute Layer 1(A) = B_tile
          Keep B_tile in SRAM

  Step 2: Compute Layer 2(B_tile) = C_tile
          C_tile stays in SRAM for Layer 3

  Step 3: Write C_tile to DRAM once (after all layers)

  Memory traffic: Only A and C to/from DRAM, B never leaves on-chip.
  Savings: 2-3× DRAM bandwidth
```

### B200's Approach: TMEM and Large Register Files

B200 introduces **Tensor Memory (TMEM)** in Blackwell:

```
Traditional GPU: SM → registers → L1 → L2 → HBM

Blackwell:       SM → TMEM (fast on-chip) → registers → L1 → L2 → HBM
                 (new 256 KB per-SM barrier)

TMEM purpose: Stage matrix blocks for tensor core arrays without going to DRAM.

Example: 70B model inference on 8-GPU DGX:
  Single GPU has 192 GB HBM, but that's distributed across 8 GPUs
  Tensor cores can work on 256 KB blocks in TMEM
  Computation:  512 TFLOPS × 256 KB = low-latency operation
  Amortize DRAM access across many operations
```

### Programming Model: Explicit Tiling

Both require careful code organization:

```
CUDA-style (B200):
  __global__ void matmul_tiled(...) {
      __shared__ float A_tile[64][64];
      __shared__ float B_tile[64][64];

      // Load tiles cooperatively
      for (tile in tiles) {
          __syncthreads();
          // Load from global → shared
          A_tile[...] = A_global[...];
          B_tile[...] = B_global[...];
          __syncthreads();

          // Compute using shared
          acc += A_tile[...] * B_tile[...];
      }
  }

HVX/Adreno-style (Snapdragon):
  Qualcomm's compiler (Qualcomm AI Engine) automatically extracts tiling opportunities.
  Developers don't write __shared__ explicitly; compiler infers tile sizes.
```

---

## Chapter 53.9: Prefetching and Lookahead — Hiding Latency with Data Movement

Even with the best caching and tiling, some loads still hit DRAM or L2, incurring 100-400 cycle latencies. GPUs hide this with **prefetching**.

### How Prefetching Works (Synchronously)

Synchronous prefetch: Request data **before** the ALU needs it.

```
Timeline without prefetch:
  Cycle 0:  ALU needs A[i]
  Cycle 1:  [stalled, waiting for DRAM]
  ...
  Cycle 100: A[i] arrives
  Cycle 101: ALU uses A[i]
  Cycle 102: ALU continues

Cost: 100 wasted cycles

Timeline with prefetch:
  Cycle 0:  Issue prefetch for A[i] (0-cycle overhead)
  Cycle 1:  ALU works on independent code (B operations, etc.)
  ...
  Cycle 100: A[i] arrives, sits in cache
  Cycle 101: ALU uses A[i] (no stall!)

Cost: 0 wasted cycles (if you have independent work)
```

### Hardware Prefetchers vs Software Prefetching

**Hardware Prefetcher** (automatic):
```
Mechanism: Track memory access patterns, predict next access.

Example: Stride-1 detection
  Accesses to A[0], A[1], A[2], ...
  Hardware detects: stride = 1
  Prefetcher automatically loads A[3], A[4], etc. before ALU requests

Limitation: Only detects simple patterns. Complex strides or data-dependent access patterns defeat it.
```

**Software Prefetching** (programmer or compiler):
```
Explicitly issue prefetch instructions:

  for (int i = 0; i < N-2; i++) {
      prefetch(A[i+2]);  // Ask for A[i+2] right now
      compute(A[i]);     // Use A[i] (already cached from prev iteration)
  }

Advantage: Can handle any pattern (compiler analyzes loop)
Disadvantage: Requires compiler support (Qualcomm Snapdragon compiler has this)
```

### Prefetching in LLM Inference

For inference, prefetching is critical because weights follow a **predictable pattern**:

```
Attention block:
  for head in num_heads:
      W_q = weight_table[5000 + head]    ← predictable sequence
      W_k = weight_table[5001 + head]
      W_v = weight_table[5002 + head]

Hardware/software prefetcher:
  See first access to 5000 → predict next is 5001, 5002, ...
  Load 5001, 5002 ASAP (they'll miss if not prefetched)

Result: 100-cycle latencies hidden. ALU sees no stall.
```

### DMA (Direct Memory Access) Overlap

For larger tiles, **DMA engines** prefetch entire blocks:

```
DMA-based prefetch:

Timeline:
  Cycle 0:   CPU issues DMA: "load 1MB block into SRAM"
  Cycle 1-5: CPU/GPU continues computing (independent data)
  Cycle 100: DMA completes, data ready in SRAM

DMA cost: ~5 cycles (issue overhead)
DRAM cost: ~100 cycles
Total: 100 cycles (compute overlapped with DMA!)
```

### Snapdragon's Hexagon DSP DMA vs Adreno

```
Snapdragon has two independent datapaths:

Hexagon Scalar DSP:
  - Has DMA engine (independent from compute)
  - Can prefetch while scalar/tensor computes
  - Typical use: load next token while computing current

Adreno GPU:
  - Load/Store units do prefetching implicitly
  - Can issue multiple outstanding requests
  - Bandwidth shared with other accesses
```

---



## Chapter 53.10: Batch Size, Wave Residency, and Register File Explosion

The relationship between batch size, wave scheduling, and register file size is **non-negotiable physics** that few discussions make explicit.

### Why Register Files Are So Large

In Chapter 36, we saw that Adreno must keep 40 waves resident to hide DRAM latency. Each wave has **its own register file**.

```
Memory latency math:
  DRAM latency: ~100-150 cycles
  Compute per wave: ~1-10 cycles per instruction
  To keep ALUs busy while waiting:
    Required waves = DRAM_latency / compute_per_wave
                  ≈ 100 / 2 ≈ 50 waves needed

Real (conservative): 40 resident waves minimum.

Register file per Compute Unit (Snapdragon Adreno 830):
  64 registers per fiber (thread)
  × 64 fibers per wave
  × 40 resident waves
  = 163,840 registers

At 4 bytes per register: 654 KB of SRAM per CU just for registers.

Adreno 830: 12 CUs × 654 KB = ~7.8 MB of registers (per CU!)
This is MASSIVE area and power.
```

### Why Decode Batch Size Matters for Inference

For inference, batch size directly determines how many waves can be active:

```
Decode (batch=1, one token per step):
  - Input: 1 sequence
  - Work parallelism: Hidden_dim / SIMD_width ≈ 4096 / 64 = 64 waves
  - Register file pressure: 64 waves × 656 KB needed
  - Can achieve: Yes, easily

Decode (batch=8, eight independent sequences):
  - Input: 8 sequences
  - Work parallelism: 8 × 64 = 512 waves
  - Register file pressure: 512 waves × 656 KB = 340 MB
  - Can achieve: NO. Too many registers needed.

Physical limitation: Can only keep 40 resident waves active per CU.
So batch > 5-10 is constrained by register file size, not bandwidth!
```

### Inference Workload Registration Pressure

```
┌────────────────────────────────────────────────────────────────┐
│ As inference batch size increases:                             │
│                                                                │
│ Batch  Register Needed  Register Available  Status             │
│ ───────────────────────────────────────────────────────        │
│   1    64 KB            656 KB              ✓ OK               │
│   4    256 KB           656 KB              ✓ OK               │
│   8    512 KB           656 KB              ✓ barely OK        │
│  16    1 MB             656 KB              ✗ SPILL to DRAM    │
│  32    2 MB             656 KB              ✗ BAD (60% DRAM)   │
│                                                                │
│ Beyond batch=8, inference starts **spilling registers to      │
│ on-chip SRAM** or worse, DRAM. This destroys latency hiding. │
└────────────────────────────────────────────────────────────────┘
```

### Why GPU Datacenters Use Batch > 32

For training or large prefill:
- Register spill is acceptable (you're computing large matrices anyway)
- Throughput (samples/sec) matters more than latency

For inference (especially decode):
- Latency is critical
- Batch size must stay small to keep waves resident
- This is why inference is fundamentally different from training

---

## Chapter 53.11: Sustained vs Burst Performance — The Thermal Ceiling

"70+ tok/s" claims for Snapdragon are **almost certainly burst numbers**, not sustained. Understanding why requires understanding thermal constraints.

### Power Budget and Thermal Envelope

Snapdragon 8 Elite sustained thermal budget (per Qualcomm/Samsung):

```
Galaxy S25+:
  Sustained thermal: 4-5W (continuous inference without throttling)
  Burst peak:        8-10W (for ~30-60 seconds before throttling)

This is a **design constraint**. The device will throttle clocks to stay under 5W.
```

### Sustained vs Burst Performance Curves

As power increases, thermal throttling reduces clock speed:

```
Clock Speed (GHz) vs Power (W)

3.5 GHz  ┌──────────────────────────────────────┐  @ max sustained
         │ full speed                           │  5W
         │                                      │
2.5 GHz  │                          ╱───────────┤  throttle kicks in
         │                        ╱
1.5 GHz  │                      ╱
         │                    ╱
0.5 GHz  │──────────────────────────────────────┴──► Power (W)
         0                                     10

Implication:
  Burst (0-60s): Can run at 3.5 GHz, use 8W → high performance
  Sustained:    Must run at ~2.5 GHz, use 5W → 40% lower performance

For LLM inference (which sustains for minutes), you get sustained performance.
```

### How This Affects "70+ tok/s" Claim

```
If 70 tok/s is burst performance at 8W:
  Naive linear: 70 × (5/8) ≈ 44 tok/s  ← WRONG assumption
  
  Why linear scaling fails:
    Power = C × V² × f  (dynamic CMOS power)
    To reduce power by 37.5% (8W→5W), you reduce voltage AND frequency
    V² × f reduction is nonlinear: halving frequency alone saves ~50% power
    but the chip doesn't linearly trade power for throughput
  
  More realistic scaling at 5W sustained:
    Clock reduction: ~25–35% (from voltage/frequency curve)
    Throughput reduction: ~25–35% (assuming BW-bound, scales with clock)
    Sustained estimate: 70 × 0.68 ≈ 47–48 tok/s (rough upper bound)
  
  Additionally: memory bandwidth also scales with NPU clock × memory controller
  At reduced DRAM frequency, BW ceiling drops proportionally.
  Sustained real-world is likely 30–50 tok/s, not 44 tok/s.

If 70 tok/s is with all cores (CPU+GPU+NPU):
  But inference mainly uses NPU, which has independent power rail
  CPU+GPU might be idle, so full chip power matters less
  But sustained, NPU alone: likely ~3-4W on its power rail

The claim becomes much more honest:
  "70+ tok/s for burst prefill with full compute engagement"
  vs
  "~40 tok/s sustained decode with reasonable thermal constraint"
```

### Power-Performance Tradeoff for Inference

```
Performance vs Power for 7B INT4 LLM on Snapdragon:

Scenario 1: Burst (60 seconds)
  Clock:     3.5 GHz
  Power:     ~8W
  Token/sec: ~70 (claimed)
  Thermal:   Unsustainable

Scenario 2: Sustained, no throttle
  Clock:     2.5 GHz
  Power:     ~5W
  Token/sec: ~50
  Thermal:   Safe indefinitely

Scenario 3: Aggressive throttle
  Clock:     1.8 GHz
  Power:     ~3W
  Token/sec: ~36
  Thermal:   Very safe, battery lasts longer

User preference depends on latency vs battery trade-off.
```

### Implications for Edge Inference

The Snapdragon design makes an **implicit trade-off**:

```
NOT optimized for:
  - Sustained high throughput (cloud would use B200)
  - Peak performance regardless of power (gaming uses GPU)

Optimized for:
  - Reasonable latency (~50 tok/s for decode)
  - Low sustained power (~4-5W continuous)
  - Long battery life in mobile phones
  - Ability to burst briefly for responsive interaction

This is the RIGHT choice for a mobile SoC.
Acknowledging this is the first step to honest specs.
```

---

## Chapter 54: B200 Architecture Overview

From NVIDIA's public materials:

- **Multi-die packaging:** Two reticle-limit dies connected by high-bandwidth on-package link
- **208 billion transistors** total
- **FP4 native support** via Transformer Engine (2 bits narrower than FP8;
  uses on-the-fly dequantization to FP8/FP16 before MAC operations —
  not simply "half the bytes": requires scaling factors and careful calibration)
- **HBM3e:** ~8 TB/s per GPU bandwidth
- **NVLink 5:** 1.8 TB/s bidirectional per GPU for multi-GPU scaling
- **Dedicated decompression engine**
- **Tensor Memory (TMEM):** New memory structure optimized for feeding tensor cores

Academic microbenchmark work on Blackwell explicitly calls out TMEM, the decompression engine, and dual-die architecture as significant changes affecting algorithm design.

---

## Chapter 55: The Constants That Change Everything

| Metric | tiny-gpu | Snapdragon 8 Elite | NVIDIA B200 |
|--------|----------|-------------------|-------------|
| Process | 130nm | 3nm (TSMC N3E) | 4nm (TSMC N4) |
| Transistors | ~100K | ~20 billion | ~208 billion |
| Compute cores | 2 | 12 CUs | 192 SMs |
| Threads | 8 | Millions | Millions |
| ALU precision | 8-bit | FP32/FP16/INT8/INT4 | FP64/FP32/FP16/FP8/FP4 |
| Memory | 256 bytes | 12 GB LPDDR5X | 192 GB HBM3e |
| Bandwidth | ~32 bits/cycle | ~84.8 GB/s | ~8 TB/s |
| Cache | None | 44+ MB on-die | ~50+ MB |
| Power | N/A | ~5W sustained | ~1000W TDP |
| Instructions | 11 | Thousands | Thousands |
| Latency hiding | None (stall) | Wave scheduling | Warp scheduling |
| Coalescing | None | Hardware | Hardware |

The physics is identical. The budgets are wildly different.

---

# Part X: Complete Reference

## Chapter 56: Snapdragon 8 Elite Full Architecture

### CPU: Oryon Gen 2

```
2× Prime cores @ 4.47 GHz (L1I: 192KB, L1D: 128KB, shared L2: 12MB)
6× Performance cores @ 3.53 GHz (L1I: 128KB, L1D: 64KB, shared L2: 12MB)
Total L2: 24 MB
ISA: ARMv8.7-A
8-wide decode, 500+ entry ROB, 6 INT + 4 FP units
```

### GPU: Adreno 830

```
3 slices × 4 CU = 12 Compute Units
1,536 FP32 shaders @ 1.2 GHz
~3.7 TFLOPS FP32, ~7.4 TFLOPS FP16
12 MB GPU L2 cache (4 MB per slice)
HW Ray Tracing Gen 2, IMR mode
Vulkan 1.3, OpenCL 3.0, DX12 FL12_1
Adreno HPM: 18 MB dedicated memory cache
UBWC v6 bandwidth compression
```

### NPU: Hexagon

```
8 scalar cores + 6 vector cores (HVX, 1024-bit SIMD) + Tensor Accelerator
~75 TOPS INT8, ~150 TOPS INT4
INT2/4/8/16, FP8/16, mixed precision
Tensor accelerator: independent power rail, DVFS
Micro Tile Inferencing
Hexagon Direct Link (zero-copy ISP → NPU path)
LPAI subsystem (always-on eNPU + DSP)
```

### Memory

```
LPDDR5X: 4 channels × 16-bit, 9600 MT/s, ~84.8 GB/s peak
12 GB (Galaxy S25+), PoP stacked on SoC
System Level Cache: 8 MB shared across all IP blocks
Total known on-die cache: ~44 MB
```

### Modem: Snapdragon X80

```
5G NR sub-6 + mmWave, 10 Gbps DL, 3.5 Gbps UL
NB-NTN satellite connectivity
5G AI Processor Gen 2
```

### Connectivity: FastConnect 7900

```
Wi-Fi 7 (802.11be): 5.8 Gbps, tri-band, 320 MHz, 4K QAM, MLO
Bluetooth 6.0, UWB, NFC
```

### Security: SPU

```
ARM SC-300 core (not Spectre/Meltdown vulnerable)
Crypto accelerator (AES-GCM, ECDSA, RSA, SHA-2)
True RNG, side-channel resistant
EAL4+, FIPS 140-2 L2
TrustZone + QSEE
```

### Power Management

```
PM8750 PMIC: Independent rails per IP block
Per-slice GPU power gating
Independent NPU tensor power rail
Sustained: ~4-5W, Peak burst: ~8-10W
Skin temperature limit: ~43°C
Samsung vapor chamber cooling (10% larger than S24+)
```

---

## Chapter 57: The Complete Mapping Table — Module by Module

Every tiny-gpu SystemVerilog module maps directly to a real GPU component. Here's the exhaustive mapping:

| tiny-gpu file | Lines | Implements | Adreno 830 | Hexagon NPU | B200 |
|---|---|---|---|---|---|
| `dcr.sv` | ~50 | Device Control Register (thread count) | Command Processor registers | QNN config | CUDA device config |
| `dispatch.sv` | ~80 | Dispatcher: assign blocks to cores | Workgroup scheduler + load balancing | Firmware tiling scheduler | GigaThread engine |
| `core.sv` | ~150 | Compute core: instantiate threads | Compute Unit (CU): 64 fibers | Scalar/Vector/Tensor unit | Streaming Multiprocessor |
| `scheduler.sv` | ~120 | 7-state FSM: fetch→decode→wait→exec→update | Wave scheduler (40-wave ready queue) | Dataflow/systolic scheduler | Warp scheduler |
| `fetcher.sv` | ~80 | Instruction fetch from program memory | Instruction cache + fetch unit per CU | Fixed-function microcode | Instruction cache + L1I |
| `decoder.sv` | ~100 | Instruction decode → control signals | Instruction decode + operand routing | Tensor control decode | Instruction decode |
| `alu.sv` | ~60 | ADD/SUB/MUL/DIV arithmetic | 256 FP32 ALUs per CU (dual-issue FP32+INT32) | Systolic MAC array | Tensor cores (FP4/FP8) |
| `lsu.sv` | ~100 | Load/Store Unit with FSM | DMA engine + coalescing + L1 write-back | Shared memory + prefetch | LSU + TMEM paths |
| `registers.sv` | ~90 | Per-thread register file (16×8-bit) | ~256 KB per CU (all fibers' state) | Shared memory + scalar regs | 256 KB per SM |
| `pc.sv` | ~80 | Program counter + branch logic | Per-wave PC + divergence stack | Firmware PC/branch | Per-warp PC + stack |
| `controller.sv` | ~150 | Memory controller arbiter | NoC fabric + L2 cache | DMA controller + scratchpad | L2 crossbar + memory bridge |
| `memory.py` | ~50 | Simulated external memory (256 bytes) | 12 GB LPDDR5X (real DDR) | 12 GB LPDDR5X | 192 GB HBM3e |

---

### Module-by-Module Detailed Mapping

#### dcr.sv → Command Processor Registers

**tiny-gpu:**
```sv
reg [7:0] device_control_register;
assign thread_count = device_control_register[7:0];
```
One register. One write port. Stores `thread_count = 8`.

**Adreno 830 Command Processor:**
```
Register range: 0x0000 - 0x3FFF (16 KB of CP registers)

Key registers:
  CP_DRAW_INDIRECT              // Issue a compute dispatch
  CP_CONTEXT_REG_BUNCH          // Load context (grid dims, shader pointer)
  CP_SET_DRAW_STATE             // Configure state (precision, constant buffers)
  CP_MEM_WRITE_ADDR             // Where to write fence value

CP processes driver-written commands in a circular buffer:
  while(CP not done):
    cmd = read_from_command_buffer()
    execute(cmd)  // Dispatch workgroup, write fence, etc.
```

**Mapping:** tiny-gpu's one DCR register → Adreno's thousands of CP registers for grid dimensions, shader pointers, constant buffer descriptors, texture descriptors, etc.

---

#### dispatch.sv → Workgroup Scheduler

**tiny-gpu:**
```sv
wire [7:0] total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
//    = (8 + 4 - 1) / 4 = 2 blocks

if (blocks_dispatched < total_blocks) begin
    core_start[i] <= 1;
    core_block_id[i] <= blocks_dispatched;
    blocks_dispatched = blocks_dispatched + 1;
end
```

**Adreno 830 Workgroup Scheduler:**
```
Input: Grid dimension (num_workgroups_x, num_workgroups_y, num_workgroups_z)
Output: Assign each workgroup to a CU + allocate registers

Scheduling algorithm:
  for each workgroup in grid:
    find_least_busy_cu()      // Which CU has fewest pending workgroups?
    allocate_registers()      // Reserve register space
    push_to_cu_queue()        // Send to that CU's ready-to-issue queue

Load balancing:
  • Early dynamic scheduling (not static block→core assignment)
  • All 12 CUs can run different workgroups simultaneously
  • If one CU finishes, scheduler immediately assigns next workgroup

Power gating:
  • If only 8 threads, Slice 1 and Slice 2 (8 CUs) power-gate
  • Only Slice 0 (4 CUs) active → 3 CUs per slice × 1 = 3 CUs
  • Saves ~70% power for tiny kernels
```

**Mapping:** tiny-gpu's static "dispatch both blocks immediately" → Adreno's dynamic workgroup scheduler with load balancing across 12 CUs, per-CU register allocation, and power-gating.

---

#### core.sv → Compute Unit (CU)

**tiny-gpu:**
```sv
generate
    for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
        alu alu_instance(...);
        lsu lsu_instance(...);
        registers registers_instance(.THREAD_ID(i), ...);
        pc pc_instance(...);
    end
endgenerate
```

**Adreno 830 Compute Unit:**
```
┌─ Compute Unit ─────────────────────────────────────────────┐
│                                                             │
│  64 fibers organized as 1 wave of 64                        │
│  (or multiple waves if resident simultaneously)             │
│                                                             │
│  ┌─ Fiber 0-10  ┌─ Fiber 11-21  ┌─ Fiber 22-32            │
│  │ SP[0]        │ SP[1]         │ SP[2]                  │
│  │ ├─ ALU       │ ├─ ALU        │ ├─ ALU                 │
│  │ ├─ LSU       │ ├─ LSU        │ ├─ LSU                 │
│  │ └─ 11 × RF   │ └─ 11 × RF    │ └─ 11 × RF (256 bytes) │
│  └─────────────┘──────────────┘─────────────             │
│                                                             │
│  ┌─ Fiber 33-43 ┌─ Fiber 44-54 ┌─ Fiber 55-63            │
│  │ SP[3]        │ SP[4]        │ SP[5]                  │
│  │ ├─ ALU       │ ├─ ALU       │ ├─ ALU                 │
│  │ ├─ LSU       │ ├─ LSU       │ ├─ LSU                 │
│  │ └─ 11 × RF   │ └─ 11 × RF   │ └─ 9 × RF              │
│  └─────────────┘──────────────┘─────────────             │
│                                                             │
│  All 64 fibers execute the same instruction (wave scheduling) │
│  Register file total: ~256 KB per CU (enough for 40 waves)  │
└─────────────────────────────────────────────────────────────┘
```

**Mapping:** tiny-gpu's 4 threads in 1 core → Adreno's 64 fibers in 1 wave, organized as 6 Shader Processors × 10-11 fibers each. Identical execution model (all in lockstep), vastly more fibers.

---

#### scheduler.sv → Wave Scheduler (The Key Difference)

**tiny-gpu:**
```sv
WAIT: begin
    reg any_lsu_waiting = 1'b0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) begin
        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10) begin
            any_lsu_waiting = 1'b1;
            break;
        end
    end
    if (!any_lsu_waiting) core_state <= EXECUTE;
    // OTHERWISE: ENTIRE CORE STALLS. ALL ALUs IDLE.
end
```

**Adreno 830:**
```sv
WAVE_SCHEDULER: begin
    // Pick next ready wave from 40 resident waves
    current_wave <= select_ready_wave(resident_waves);

    case(wave_state[current_wave])
        WAITING_ON_MEMORY:
            // Check if memory is ready
            if (lsu_ack[current_wave]) begin
                wave_state[current_wave] <= READY;
                // Keep going to next iteration of WHILE loop
            end else begin
                // Memory not ready, don't schedule this wave
                // Check NEXT wave...
            end
        READY:
            // Issue next instruction from this wave to SPs
            issue_to_SPs(fetch_instruction(current_wave));

        DONE:
            // Free up registers, remove from resident set
            deallocate_registers(current_wave);
            // Wake up next pending workgroup if space
    endcase
end
```

**Key difference:**
- tiny-gpu: If ANY thread waits → ENTIRE CORE WAITS
- Adreno: If ANY wave waits → SKIP IT, SCHEDULE NEXT WAVE

This single architectural change is responsible for the **60-95× ALU utilization improvement**.

**Mapping:** tiny-gpu's 7-state per-core FSM → Adreno's per-wave scheduler managing 40 waves independently, switching between them in microseconds.

---

#### fetcher.sv & decoder.sv → Instruction Fetch & Decode

**tiny-gpu:**
```sv
IDLE: begin
    if (core_state == FETCH) begin
        fetcher_state <= FETCHING;
        mem_read_valid <= 1;
        mem_read_address <= current_pc;
    end
end
FETCHING: begin
    if (mem_read_ready) begin
        instruction <= mem_read_data;
        fetcher_state <= FETCHED;
    end
end
```

**Adreno 830:**
```
Instruction Fetch Path:
  1. Per-wave program counter (PC) maintained
  2. Fetch from instruction cache (16-32 KB typical per CU)
     - If hit: 1-4 cycles latency
     - If miss: 20+ cycles (fetch from L2 or further)
  3. Decode: Split 64-bit instruction into micro-ops
  4. Operand collection: Gather register file reads for all 64 fibers
  5. Issue: Send micro-ops to 6 Shader Processors

Pipelining:
  Cycle N:     Fetch instruction at PC
  Cycle N+1:   Decode (stage 2 pipeline)
  Cycle N+2:   Collect operands (register file read)
  Cycle N+3:   Execute (ALU, LSU, etc.)
  Cycle N+4:   Memory/Write-back

  Multiple instructions in flight (5-6 stage pipeline)
```

**Mapping:** tiny-gpu's 3-state sequential fetcher → Adreno's pipelined instruction fetch with I-cache and per-wave PC tracking.

---

#### alu.sv → Shader ALUs

**tiny-gpu:**
```sv
case (decoded_alu_arithmetic_mux)
    ADD: alu_out_reg <= rs + rt;
    SUB: alu_out_reg <= rs - rt;
    MUL: alu_out_reg <= rs * rt;
    DIV: alu_out_reg <= rs / rt;
endcase
```

**Adreno 830 per Compute Unit:**
```
256 FP32 ALUs (64 fibers × 4 ALUs per fiber, or dual-issue per SP)
+ 512 FP16 ALUs (higher precision in parallel)

Instruction set (vs tiny-gpu's 4 basic ops):
  • FP32/FP16 arithmetic: ADD, SUB, MUL, MAD (multiply-add)
  • Integer arithmetic: INT32 ADD/SUB/MUL/SHL/SHR, etc.
  • Bit operations: AND, OR, XOR, NOT, SHIFT, ROTATE
  • Comparison: CMP, LT, GT, EQ
  • Float operations: SQRT, RECIP, LOG, EXP
  • Transcendentals: SIN, COS, TAN (via lookup tables)
  • Special: Integer multiply-extend, bit count, bit reverse

Dual-issue capability (per SP):
  Cycle N: Issue FP32 op to FP pipe
  Cycle N: Issue INT32 op to INT pipe (simultaneously)
  Result: 256 FP32 + 256 INT32 ops per cycle
```

**Mapping:** tiny-gpu's 1 simple 4-operation ALU per thread → Adreno's massive 256-entry parallel ALU array with dual-issue pipelines and full IEEE 754 support.

---

#### lsu.sv → Load/Store Unit + Coalescing

**tiny-gpu:**
```sv
REQUESTING: begin
    mem_read_valid <= 1;
    mem_read_address <= rs;
    lsu_state <= WAITING;
end
WAITING: begin
    if (mem_read_ready) begin
        lsu_out <= mem_read_data;
        lsu_state <= DONE;
    end
end
```

**Adreno 830:**
```
Load-Store Unit (per Shader Processor, serving 11-16 fibers):

1. Coalescing Stage
   Input: 64 fiber addresses (if all loads same instruction)
   • Examine all addresses
   • Group adjacent addresses (within 128-byte window)
   • Example: addresses [0,1,2,3,8,9,10,11] → 2 requests
   • Worst case (stride-16): 64 separate requests (no coalescing)
   • Best case (stride-1): 1 request (64 bytes from one cache line)
   Output: 1-64 memory requests

2. Per-request path:
   a) Check L1 cache (64 KB, 64-byte lines)
      Hit (~4 cycles): immediate return
      Miss: continue

   b) Check L2 cache (4 MB per slice)
      Hit (~20-40 cycles): fetch from L2
      Miss: continue

   c) Check SLC (8 MB shared)
      Hit (~30-40 cycles): fetch from SLC
      Miss: continue

   d) LPDDR5X DRAM (~100-150 cycles)
      Fetch with actual DDR latency

3. Write-back caching (for stores)
   Store → L1 cache dirty bit
   Return immediately to core (cache coherent)
   L1 flushes to L2 later (asynchronously)
   Core doesn't wait for DRAM confirmation

4. Prefetch (speculative)
   Hardware can fetch next cache line (spatial locality)
   Hides latency for sequential access patterns
```

**Mapping:** tiny-gpu's serial 1-request-at-a-time LSU → Adreno's coalescing-enabled parallel LSU with 16+ in-flight requests per CU, multi-level caching, and write-back semantics.

---

#### registers.sv → Register File + Shared Memory

**tiny-gpu:**
```sv
module registers #(
    parameter THREAD_ID = 0,
)
registers[13] <= blockIdx;      // Read-only
registers[14] <= blockDim;      // Read-only (= 4)
registers[15] <= threadIdx;     // Read-only (= THREAD_ID)
```

**Adreno 830:**
```
Register file: ~256 KB per CU
  • Divided into 64 register files (one per fiber)
  • Each fiber: 256 registers × 4 bytes = 1 KB
  • All resident simultaneously (unlike CPU which spills to stack)

Thread mapping:
  registers[0-12]:    User-allocated variables
  registers[13]:      Local ID X (within workgroup)
  registers[14]:      Local ID Y
  registers[15]:      Local ID Z
  registers[16-23]:   Work group ID X, Y, Z + metadata
  registers[24+]:     Reserved for compiler

Wave scheduling benefit:
  40 resident waves × 64 fibers × 256 regs = 655,360 registers
  ≈ 2.5 MB register space (vs Adreno's 256 KB per CU)

  This massive register file is the POWER COST of wave scheduling:
  # Transistors ≈ capacity × (area per bit + read/write logic)
  256 KB × ~6 transistors per bit ≈ 12,000,000 transistors (per CU!)
  = 12 CUs × 12M = 144 million transistors just for registers

  This is comparable to entire Snapdragon 8 Elite CPU L2 cache (24 MB)
  Registers are the single biggest power/area consumer per CU.

Shared Memory (LDS: 64 KB per CU):
  • Fast on-chip SRAM (5-10 cycle latency)
  • All fibers in CU can read/write same addresses
  • Synchronization barriers (__syncthreads equivalent)
  • Useful for tile reuse in matrix multiply

  tiny-gpu has NO shared memory equivalent
  (would require inter-thread communication)
```

**Mapping:** tiny-gpu's 16×8-bit per-thread register file (128 bytes total) → Adreno's 256 KB per-CU register file + 64 KB shared memory, enabling massive parallelism.

---

#### controller.sv → Memory Hierarchy + Crossbar

**tiny-gpu:**
```sv
if (channel_read_valid[j] && !channel_serving_consumer[j]) begin
    channel_serving_consumer[j] = 1;
    current_consumer[i] <= j;
    mem_read_valid[i] <= 1;
    mem_read_address[i] <= consumer_read_address[j];
    break;  // First-come, first-served by index
end
```

**Adreno 830:**
```
Memory Hierarchy:

┌─ Fiber ─────────────────────────┐
│ Registers (4-cycle)              │
└─────┬──────────────────────────┘
      │
      ▼
┌─ L1 Cache ──────────────────────┐
│ 32-64 KB per SP, fully assoc     │
│ 64-byte line, write-back         │
│ 4-cycle hit, 20+ cycle miss      │
└─────┬──────────────────────────┘
      │
      ▼
┌─ L2 Cache ──────────────────────┐
│ 4 MB per slice, shared by 4 CUs  │
│ 20-40 cycle latency              │
└─────┬──────────────────────────┘
      │
      ▼
┌─ SLC (System-Level Cache) ──────┐
│ 8 MB shared across all IP blocks │
│ 30-40 cycle latency              │
└─────┬──────────────────────────┘
      │
      ▼
┌─ LPDDR5X Memory Controller ─────┐
│ 4 channels × 16-bit × 9600 MT/s  │
│ ~84.8 GB/s peak                  │
│ QoS arbitration (priority levels) │
│ 100-150 cycles to DRAM           │
└─────┬──────────────────────────┘
      │
      ▼
┌─ LPDDR5X DRAM ──────────────────┐
│ 12 GB capacity                   │
│ ~100-150 ns (100-150 cycles)     │
└──────────────────────────────────┘

Arbiter behavior (vs tiny-gpu's fixed priority):
  • QoS-aware: priority levels per IP block
  • Display pipeline: highest priority (120 Hz deadline)
  • ISP (camera): second priority (real-time)
  • CPU/GPU/NPU: fair sharing of remaining bandwidth
  • Credit-based flow control: prevent starvation

Coalescing window (per request):
  Up to 64 fiber addresses merged per cache line (64 bytes)
  Depending on stride pattern, reduces traffic by 4-64×
```

**Mapping:** tiny-gpu's simple 4-channel priority arbiter → Adreno's QoS-aware NoC fabric with 4-level cache hierarchy, coalescing, write-back semantics, and credit-based fairness.

---

## Chapter 57.5: Complete Picture — What Adreno Adds and Why

The architectural gap between tiny-gpu and Adreno 830 can be summarized as **6 categories of improvements**, each solving a specific tiny-gpu bottleneck:

### 1. Wave Scheduling (solves: core stalls on memory)

**Problem:** tiny-gpu's scheduler.sv WAIT state freezes all ALUs when ANY thread waits.
**Solution:** Adreno keeps 40 waves resident, switches to next ready wave when current wave stalls.
**Impact:** 60-95× better ALU utilization, ~20-60 visible cycles instead of 120+ cycles per kernel.

### 2. Memory Coalescing (solves: no coalescing)

**Problem:** tiny-gpu sends 8 separate requests for 8 addresses.
**Solution:** Adreno's coalescing unit merges adjacent addresses into single cache-line requests.
**Impact:** 4-64× fewer memory transactions, bandwidth savings especially for stride-1 patterns.

### 3. Multi-Level Caching (solves: no data cache)

**Problem:** tiny-gpu sends every load to external memory, second load costs same as first.
**Solution:** Adreno has L1 → L2 → SLC → DRAM hierarchy; repeated accesses hit L1 in ~4 cycles.
**Impact:** 5-25× faster for repeated/nearby accesses, activations cached across layers.

### 4. Instruction Caching (solves: instruction fetch bottleneck)

**Problem:** tiny-gpu fetches every instruction through 1-channel program memory controller.
**Solution:** Adreno has per-CU instruction cache; once cached, fetches are 1-4 cycles.
**Impact:** 13× fewer external instruction fetches, especially in tight loops (normal for inference).

### 5. Register File Capacity (enables wave scheduling)

**Problem:** tiny-gpu's 128-byte total register space can't hold 40 waves' state.
**Solution:** Adreno dedicates 256 KB per CU (144 million transistors) to support resident waves.
**Impact:** Prerequisite for wave scheduling; massive power/area cost but worth it for latency hiding.

### 6. Per-Fiber Independent Execution (improves branch divergence handling)

**Problem:** tiny-gpu assumes all threads branch to same PC (TODO: branch divergence).
**Solution:** Adreno maintains per-wave divergence stack; some fibers can branch independently while others continue.
**Impact:** Enables complex control flow (if-else within kernels), though with serialization cost when divergent.

---

### Module-by-Module Improvement Summary

| Module | tiny-gpu Limitation | Adreno 830 Solution | Transistor Cost |
|--------|---|---|---|
| `scheduler.sv` | WAIT stalls core | Wave scheduling (40 resident) | ~100K (control logic) |
| `fetcher.sv` | All fetches → ext mem | Instruction cache (16-32 KB per CU) | ~50K per CU × 12 = 600K |
| `lsu.sv` | No coalescing, no cache | Coalescing unit + L1 cache (32-64 KB per SP) | ~200K per CU × 12 = 2.4M |
| `controller.sv` | Fixed-priority arbiter | NoC + L2 cache (4 MB per slice) + SLC (8 MB) | ~5M (L2 + SLC cache) |
| `registers.sv` | 128 bytes total | 256 KB per CU | ~144M (largest per-CU component!) |
| `alu.sv` | 4 ops, 8-bit | 1,536 parallel ALUs, FP32/FP16/INT32 | ~50M (ALU array) |
| **Total** | ~100K transistors | ~20 billion transistors | ~200× increase |

The surprising insight: **Wave scheduling (the biggest performance win) is enabled by massive register files**, which are the single biggest transistor consumer on the chip. This is why GPU register files are so large and so power-hungry.

---



| tiny-gpu file | What it does | Adreno 830 | Hexagon NPU | B200 |
|---------------|-------------|------------|-------------|------|
| `dcr.sv` | Stores thread count | Command Processor registers | QNN runtime config | CUDA driver config |
| `dispatch.sv` | Assigns blocks to cores | Workgroup scheduler | Firmware tile scheduler | GigaThread Engine |
| `core.sv` | Processes one block | Compute Unit (CU) | Scalar/Vector/Tensor unit | Streaming Multiprocessor |
| `scheduler.sv` | FSM: fetch→execute→update | Wave scheduler (multi-wave) | Dataflow/systolic scheduling | Warp scheduler |
| `fetcher.sv` | Gets instruction from memory | Instruction fetch + I-cache | Fixed-function microcode | Instruction fetch + I-cache |
| `decoder.sv` | Instruction → control signals | Instruction decode unit | Tensor accelerator control | Instruction decode |
| `alu.sv` | ADD/SUB/MUL/DIV | 256 FP32 ALUs per CU | Systolic MAC array | Tensor Cores + CUDA cores |
| `lsu.sv` | Async load/store | LSU + coalescing | DMA + shared scratchpad | LSU + coalescing + TMEM |
| `registers.sv` | 16×8-bit per thread | ~256 KB register file per CU | Shared memory | 256 KB register file per SM |
| `pc.sv` | Program counter + branch | PC + divergence stack | Firmware control flow | PC + divergence stack |
| `controller.sv` | Arbitrates memory access | Memory crossbar + L2 + SLC | DMA controller | L2 + crossbar + HBM controller |
| `memory.py` | Flat external memory | 12 GB LPDDR5X (banked, cached) | 12 GB LPDDR5X | 192 GB HBM3e |

---

## Chapter 58: The Complete Architecture Mapping

Here's how every module in tiny-gpu maps to real hardware, with the core insight running through:

```
┌─────────────────────────┬────────────────────────────┬────────────────────────────────┐
│  tiny-gpu               │  Snapdragon 8 Elite        │  NVIDIA B200                   │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  setup.py (driver)      │  QNN runtime / LiteRT      │  CUDA driver + TensorRT        │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  dispatch.sv            │  Adreno 830 workgroup      │  GigaThread Engine             │
│  (block → core)         │  scheduler + power-gating  │  (CTA → SM)                    │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  scheduler.sv           │  Wave scheduler            │  Warp scheduler                │
│  WAIT state (stall)     │  → switch wave, hide stall │  → switch warp, hide stall     │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  controller.sv          │  NoC fabric (QoS-aware)    │  L2 crossbar                   │
│  (no coalescing,        │  + coalescing              │  + coalescing + TMEM           │
│  fixed-priority)        │  + SLC shared cache        │  + L2 cache                    │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  lsu.sv                 │  DMA engine                │  Load/store units              │
│  (one req at a time)    │  (prefetch + double buffer) │  (thousands in flight/SM)     │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  alu.sv                 │  Hexagon NPU systolic MAC  │  Tensor cores (FP4/FP8)        │
│  (1 op, cheap)          │  (~75 TOPS INT8)           │  (~2,500 TOPS FP8)             │
├─────────────────────────┼────────────────────────────┼────────────────────────────────┤
│  registers.sv           │  Register file + NPU       │  256 KB register file / SM     │
│  (128 bytes total)      │  shared scratchpad         │  (~49 MB across all SMs)       │
└─────────────────────────┴────────────────────────────┴────────────────────────────────┘

Core insight running through every row:
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Moving data costs ~170× more energy than computing on it.                          │
│  Every architectural feature above exists to reuse data already on-chip             │
│  rather than paying that DRAM cost again.                                            │
│  tiny-gpu pays the full DRAM cost every time — and achieves 0.8% ALU utilization.  │
│  Real chips spend billions of transistors to avoid that cost.                        │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Chapter 58.5: Electrical Engineering — From Verilog to Silicon

### What tiny-gpu's SystemVerilog Actually Describes

Every line of SystemVerilog **describes a physical circuit** made of transistors. Here's the translation chain:

```
Verilog              Circuit Diagram            Transistors           Silicon
─────────────────────────────────────────────────────────────────────────────

assign c =          ┌─────────────┐
a + b;              │   Full-Add  │           ~40-80 transistors
                    │   Adder     │
                    └─────────────┘

case (opcode)   →   ┌──────────────────┐
    ADD: ...        │  4-to-1 Mux      │       ~30 transistors (select logic)
    SUB: ...        │  (select opcode) │
    ...             └──────────────────┘

reg [7:0] out;  →   ┌──────────────────┐
                    │  8× Flip-Flops   │       ~48 transistors (6 per FF)
                    └──────────────────┘
```

### Example: alu.sv → Physical Transistor Count

From the tiny-gpu ALU in `alu.sv`:

```sv
case (decoded_alu_arithmetic_mux)
    ADD: alu_out_reg <= rs + rt;   // 8-bit addition
    SUB: alu_out_reg <= rs - rt;   // 8-bit subtraction
    MUL: alu_out_reg <= rs * rt;   // 8-bit multiplication
    DIV: alu_out_reg <= rs / rt;   // 8-bit division
endcase
```

**Transistor breakdown:**

```
Operation          Implementation              Transistor Count    Notes
───────────────────────────────────────────────────────────────────────────
rs + rt            8-bit ripple-carry adder    ~40-80 transistors  Ripple = simple, slow
                   (or carry-lookahead)        ~100-200 tx (fast)  CLA faster but larger

rs - rt            8-bit subtractor            ~40-80 transistors  Subtractor ≈ adder + inverter
                   (adder with inverted input)

rs * rt            8×8 array multiplier        ~200-500 tx         Array = basic O(n²)
                   (or Booth multiplier)       ~400-800 tx (Booth) Booth = more complex

rs / rt            8-bit divider               ~500-1,500 tx       Restoring or non-restoring
                   (restoring algorithm)                           Dividers are large!

4-to-1 mux         Select which result        ~30 transistors      Simple logic
(select opcode)

8-bit register     8 flip-flops storing       ~48 transistors      6 transistors per flip-flop
(alu_out_reg)      the result                 (D-latch style)

────────────────────────────────────────────────────────────────────────────
ONE ALU Total      (ripple ADD + sub + mult)   ~800-1,700 tx       Conservative lower bound
```

**For all of tiny-gpu:**

```
tiny-gpu Component                      Count       Transistors Each  Total
─────────────────────────────────────────────────────────────────────────────
ALU (alu.sv)                           8           ~800-1,700        ~6,400-13,600
Register file (registers.sv)           8 threads × 16 regs × 8-bit
                                                    ~48 × 16 × 8      ~6,144 tx per thread
                                                    × 8 threads       ~49,152
PC + Decoder (pc.sv + decoder.sv)                  ~200-500           ~2,000
LSU (lsu.sv)                           8           ~200-300           ~2,400
Fetcher (fetcher.sv)                   2           ~300-400           ~800
Scheduler (scheduler.sv)               2           ~1,000 (FSM)       ~2,000
Multiplexers, wiring, control          various     ~10,000-20,000
─────────────────────────────────────────────────────────────────────────────
TOTAL tiny-gpu estimate                                              ~80,000-110,000 tx

For comparison:
  Intel 4004 (1971):                   ~2,300 transistors (first CPU!)
  tiny-gpu:                            ~100,000 transistors
  Snapdragon 8 Elite (2024):           ~20,000,000,000 transistors (200,000× larger)
```

### GDS Files: From SystemVerilog to Manufacturing

tiny-gpu includes **GDS files** — the actual physical layouts:

```
Repository structure:
  gds/0/gpu.gds        ← Final physical layout, ready for fab
  gds/1/gpu.gds        ← Alternative implementation (2 variants)
```

GDS (Graphic Data System) is binary format describing the exact polygon coordinates of every metal trace, via, and transistor on the chip. A foundry reads this and:

1. **Reticle alignment:** Splits the design into reticle tiles (if larger than ~26mm × 33mm)
2. **Mask generation:** Creates optical masks for each manufacturing step
3. **Wafer processing:** Uses masks to etch/deposit material layer by layer
4. **Verification:** E-beam inspection to catch defects before shipping

For a 130nm process (Sky130), the manufacturing steps include:

```
Step 1: Grow silicon dioxide (SiO₂) insulator on wafer
Step 2: Deposit polysilicon (transistor gates)
Step 3: Deposit metal layers (M1, M2, M3, M4, M5)
Step 4: Deposit vias (connections between layers)
Step 5: Repeat with photolithography masks
...
~50+ steps total for a modern process
```

### Manufacturing Flow: SystemVerilog → GDS

```
┌─────────────────────────────────────────────────────────────┐
│  SystemVerilog Source Code                                  │
│  (src/*.sv files from tiny-gpu GitHub)                      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  sv2v (SystemVerilog to Verilog Converter)                  │
│  • Removes features unsupported in basic Verilog            │
│  • Expands generate blocks into explicit instances          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Yosys (Open-Source Synthesizer)                            │
│  • Convert behavioral Verilog to logic gates                │
│  • Optimize gate count (reduce redundancy)                  │
│  • Technology mapping (target 130nm standard cells)          │
│  • Output: netlist (gates + wires)                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  OpenLane (Open-Source Place & Route)                       │
│  • Floor planning: where to put which blocks                │
│  • Placement: arrange cells on die to minimize wirelength    │
│  • Clock tree synthesis: distribute clock signal            │
│  • Routing: draw metal wires connecting all cells           │
│  • DFM (Design for Manufacturability) checks                │
│  • Output: DEF (Design Exchange Format)                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  GDS Generator (from DEF)                                   │
│  • Convert DEF layout to GDS binary format                  │
│  • Add process-specific layers (vias, metal)                │
│  • SkyWater 130nm PDK (Process Design Kit) fills in details  │
│  • Output: gds/0/gpu.gds and gds/1/gpu.gds                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Tiny Tapeout Program (at SkyWater 130nm)                   │
│  • Combines 100s of designs onto one shuttle wafer          │
│  • Cost per designer: ~$500-1,000 (shared cost)             │
│  • Typical turnaround: 6-9 months                           │
│  • Designs get real silicon die + assembled package         │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Manufacturing (SkyWater Foundry)                           │
│  • Photolithography (UV masks project pattern onto wafer)    │
│  • Etching (plasma removes material through mask patterns)   │
│  • Deposition (CVD adds metal, oxide, polysilicon layers)   │
│  • ~50+ process steps to build all layers                   │
│  • Wafer testing (continuity, leakage, delay)               │
│  • Die cutting and packaging                                │
│  • Final testing (functional validation)                    │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│  Physical Silicon Chip                                       │
│  • ~100,000 transistors on a tiny die                       │
│  • ~2mm × 2mm die size (estimate)                           │
│  • 40-pin DIP package (DIL breadboard-friendly)             │
│  • Can be programmed and tested on real hardware            │
└─────────────────────────────────────────────────────────────┘
```

### Tiny Tapeout Program

The **Tiny Tapeout** program is a remarkable initiative:

- **What it is:** A community shuttle service bundling hundreds of small open-source designs onto one manufacturing run
- **Cost:** ~$500-1,000 per design (shared foundry costs across ~150 designs)
- **Traditional cost:** $50,000-500,000+ for individual design (non-recurring engineering fees)
- **Process:** SkyWater 130nm (open PDK, no NDA required)
- **Timeline:** 6-9 months from submission to silicon in hand
- **Community:** 4,000+ designers, growing

tiny-gpu was likely submitted as part of **Tiny Tapeout 7 or 8** (2024). The GDS files in the repo are the actual layouts that were sent to the foundry.

### Process Node Comparison: 130nm vs 3nm Transistor Density

```
Process Node    Transistor Density    Year    Platform           Notes
─────────────────────────────────────────────────────────────────────────────
130 nm          ~10,000 tx/mm²        2001    Old desktop CPUs   SkyWater 130nm
                                              Pentium III era    (open PDK)

65 nm           ~50,000 tx/mm²        2005    Early-mid market   Core 2 Duo
45 nm           ~150,000 tx/mm²       2007    High-volume        i7 generation

28 nm           ~500,000 tx/mm²       2011    Kepler (NVIDIA)    Snapdragon 820

14 nm           ~2,000,000 tx/mm²     2015    Skylake (Intel)    Snapdragon 835
7 nm            ~16,000,000 tx/mm²    2019    A12 Bionic         Apple, TSMC

5 nm            ~32,000,000 tx/mm²    2021    A14/A15, S21       Apple, Samsung

3 nm            ~100,000,000 tx/mm²   2024    Snapdragon 8 Elite TSMC N3E

Scaling insight: 130nm → 3nm = 43× smaller linear dimension
              → 43² = 1,850× higher transistor density
              → tiny-gpu at 130nm = ~100K tx in ~10mm²
              → same design at 3nm = ~185M transistors in 0.3mm²
```

### Key Implication: Scaling tiny-gpu to Modern Process

If tiny-gpu (100,000 transistors) were implemented at modern nodes:

```
Process    Die Size    Power (at same voltage)
──────────────────────────────────────────────
130 nm     ~10 mm²     ~5W (current)
28 nm      ~0.3 mm²    ~2W (benefits from lower capacitance)
3 nm       ~0.005mm²   ~0.1-0.5W (much lower C,V, better P = CV²f)
```

At 3nm with modern operating techniques, a 100K-transistor design would be **nearly unmeasurable** — it would be smaller than the test pads! This illustrates why modern chips use billions of transistors — only then do they achieve meaningful density and performance.

---



### Hardware-First Exercises

**Exercise 1: Bandwidth-limited token bound**

For any platform, compute:

```
tokens/sec ≤ DRAM_bandwidth / bytes_of_weights_per_token
```

Use this to sanity-check any tok/s claim. Tie to Roofline: if operational intensity is low, bandwidth dominates.

**Exercise 2: Energy-per-bit intuition**

Assume DRAM is ~100× more expensive than SRAM and ~1000× more expensive than simple ALU ops. Then ask: which design choices reduce DRAM traffic?

**Exercise 3: Interpret Qualcomm's claims through physics**

- "Microtile inferencing eliminates inter-layer traffic" → activation bandwidth reduction
- "Native INT4 improves bandwidth efficiency" → fewer bytes per weight moved
- "Shared memory concurrency support" → more on-chip reuse, less DRAM bouncing
- "Dedicated power rails" → independent DVFS for tensor vs scalar/vector

### Software Stack Mapping

- **QNN / AI Engine Direct** is the unified backend that frameworks target
- Many high-performance paths require **quantized models** (HTP backend constraints)
- **NNAPI is deprecated** in Android 15; modern deployment uses LiteRT QNN integration
- The compiler pipeline: tiling for scratchpad → fusion → vectorization → DMA → tensor microkernels

---

## Chapter 41.5: Software Stack Complete Path

### tiny-gpu: Hand-Encoding to Execution

tiny-gpu requires **hand-written assembly code** (or Python-generated bitstreams) and manual memory management:

```
┌─────────────────────────────────────────────────────┐
│ Step 1: Write Python kernel code                    │
│                                                     │
│ def matadd(A, B, C, n):                            │
│     for i in range(n):                             │
│         C[i] = A[i] + B[i]                         │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: Hand-encode to tiny-gpu ISA (11 ops)       │
│                                                     │
│ Instruction memory:                                 │
│   0: LDR  $0, 0($1)    # Load A[tid] into r0      │
│   1: LDR  $2, 4($1)    # Load B[tid] into r2      │
│   2: ADD  $3, $0, $2   # r3 = r0 + r2             │
│   3: STR  $3, 8($1)    # Store r3 into C[tid]     │
│   4: BRZ  $4, -4       # If tid != 0, exit        │
│   ...                                               │
│                                                     │
│ Data memory:                                        │
│   0x00-0x07: A[0..7]    (8 bytes)                  │
│   0x08-0x0f: B[0..7]    (8 bytes)                  │
│   0x10-0x17: C[0..7]    (8 bytes)                  │
│   0x18-0x1f: temp       (8 bytes)                  │
│                                                     │
│ Total: 4 instructions, 48 bytes memory              │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: Write simulator driver (Python)             │
│                                                     │
│ def run_kernel():                                   │
│     sim = Simulator()                              │
│     sim.load_program(program_bytecode)             │
│     sim.load_memory(A, B, initial_C)               │
│     for cycle in range(MAX_CYCLES):                │
│         sim.tick()                                 │
│     return sim.read_memory()                       │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Simulator executes cycle-by-cycle          │
│                                                     │
│ Cycle 1: Fetch "LDR  $0, 0($1)"                    │
│ Cycle 2: Decode, address = 0                       │
│ Cycle 3: Memory request sent                       │
│ ...                                                 │
│ Cycle N: All threads done, return results          │
│                                                     │
│ Output: C[0..7] with sum results                   │
└─────────────────────────────────────────────────────┘
```

**Characteristics of tiny-gpu pipeline:**
- **Manual code generation**: No compiler, developer writes assembly directly
- **No auto-vectorization**: Each thread manually specified
- **Memory by hand**: Developer chooses addresses and timing
- **Simulator only**: Code doesn't run on real hardware (would need to synthesize tiny-gpu first)
- **Deterministic cycle count**: Cycle-by-cycle execution is repeatable and inspectable

### Snapdragon 8 Elite: Compiler-Driven to Execution

Modern APUs use a sophisticated compiler pipeline:

```
┌─────────────────────────────────────────────────────┐
│ Step 1: High-level framework code                   │
│                                                     │
│ # PyTorch / TensorFlow / ONNX                       │
│ import torch.nn.functional as F                    │
│ import tensorrt as trt                             │
│                                                     │
│ # Convert float32 → INT4 (quantized)               │
│ model = quantize(model, bits=4)                    │
│ # Export as ONNX or TFLite                         │
│ export_tflite("model.tflite")                      │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: LiteRT / QNN Graph Compiler                 │
│                                                     │
│ Input:  ONNX/TFLite graph (abstract DAG)           │
│         model.tflite                                │
│                                                     │
│ Phase A: Graph Optimization                         │
│   • Layer fusion:   Conv3x3 + ReLU → FusedConv    │
│   • Operator removal: Dead code elimination        │
│   • Constant folding: 1.0 * x → x at compile-time │
│                                                     │
│ Phase B: Quantization (if not already done)        │
│   • Calibrate scale/zero-point per layer          │
│   • Convert FP32 weights → INT4 + scale           │
│                                                     │
│ Phase C: Mapping to QNN Ops                        │
│   • FusedConv → HTP's qnn::Conv2d                 │
│   • MatMul → HTP's qnn::MatMul or tensor ops      │
│   • Add → scalar ALU or tensor ADD                │
│                                                     │
│ Phase D: Memory Planning                           │
│   • Assign each tensor to L1/L2/SLC/DRAM          │
│   • Tile large matrices: 4KB L1 scratchpad        │
│   • Generate DMA descriptors for transfers         │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: Generate Micro-kernels & DMA Program       │
│                                                     │
│ For each tiled operation:                           │
│   Output: HTP ISA microkernel (scalar + vector)   │
│           DMA program (load weights to L1)         │
│           Scheduler commands (grid dispatch)       │
│                                                     │
│ Example: Conv2d (fused with ReLU, quantized)      │
│   • Load tile of weights (4KB) via DMA            │
│   • Load tile of input (8KB) via DMA              │
│   • Execute convolution loop on HTP (vector ALU)  │
│   • Accumulate in L2 (32KB buffer)                │
│   • Store output tile via DMA to SLC              │
│   • Repeat for next tile                          │
│                                                     │
│ Total per tile: ~100-500 HTP instructions         │
│ Runtime: ~0.1-1ms depending on tile size          │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Runtime Package (QNN Delegate)              │
│                                                     │
│ .qnn file or in-memory graph:                      │
│   • Compiled binaries (HTP microkernels)           │
│   • Quantization parameters (per-layer scales)    │
│   • DMA program (memory layout + transfers)        │
│   • Execution order (topological sort of ops)      │
│                                                     │
│ Packaged into APK or shipped separately            │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 5: Runtime on Device                          │
│                                                     │
│ Framework (PyTorch/TensorFlow) on CPU:            │
│   • Load .qnn file                                │
│   • Prepare inputs (quantize if needed)           │
│   • Invoke HTP via QNN Runtime                    │
│   • Wait for completion                           │
│   • Dequantize outputs (INT4 → FP32)              │
│                                                     │
│ QNN Runtime (msm_qnn kernel module):              │
│   • Map virtual addresses to physical IOVA        │
│   • Set up MMU for HTP memory access              │
│   • Kick off HTP firmware                         │
│   • Interrupt handler on completion               │
│                                                     │
│ HTP Firmware:                                       │
│   • Load DMA program                              │
│   • Fetch first microkernel from program memory   │
│   • Execute on vector/scalar ALUs                │
│   • Trigger DMA transfers as needed               │
│   • Repeat until program complete                 │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ Step 6: Results on GPU (if using GPU instead)      │
│                                                     │
│ Alternative to HTP:                                │
│ • Same pipeline but target Adreno 830             │
│ • Compile to Adreno ISA (very different from HTP)  │
│ • Use GPU driver (kgsl) instead of QNN runtime    │
│ • Similar DMA + kernel structure                  │
│                                                     │
│ Performance: ~50% faster on GPU vs HTP for FP32   │
│ Power: ~2× worse on GPU vs HTP for INT4 inference │
│                                                     │
│ Developer choice depends on:                       │
│   • Model size (HTP better for tiny models)       │
│   • Precision (HTP better for INT4, GPU for FP32) │
│   • Latency requirement (GPU lower latency)       │
│   • Power constraint (HTP lower power)            │
└─────────────────────────────────────────────────────┘
```

**Characteristics of Snapdragon pipeline:**
- **Automatic compilation**: Framework handles code generation
- **Auto-vectorization**: Compiler maps scalar ops to SIMD
- **Hardware memory planning**: Compiler assigns data to cache levels
- **Multi-target**: Can compile to HTP, GPU, or mixed
- **Deterministic but complex**: Performance tied to compiler version & quantization

### Key Differences: tiny-gpu vs Snapdragon Compiler Pipeline

| Aspect | tiny-gpu | Snapdragon |
|--------|----------|-----------|
| **Code format** | Hand-written ISA bytecode | High-level graph (ONNX/TFLite) |
| **Compiler** | Manual / no tool | LiteRT / SNPE / QNN framework |
| **Parallelism** | Explicit (4 threads) | Implicit (framework chooses) |
| **Memory management** | Manual addressing | Automated tiling & DMA |
| **Quantization** | Not supported | Automatic (INT4, INT8, FP16) |
| **Target hardware** | Simulator only | Real silicon (HTP or GPU) |
| **Performance tuning** | Cycle-by-cycle inspection | Compiler heuristics + profiling |
| **Development effort** | ~10 min per kernel | ~30 sec (automatic) |
| **Optimization ceiling** | Limited by hand-tuning | Compiler can find non-obvious opts |

### The Bridge: tiny-gpu as Pedagogical Model

tiny-gpu's manual pipeline teaches what the Snapdragon compiler hides:

```
Human writes tiny-gpu code
          ↓
         [manual cycle-counting]
          ↓
Understands: memory latency, stalls, utilization
          ↓
         [reads compiler output]
          ↓
Appreciates: automated scheduling, latency hiding, coalescing
          ↓
         [writes better quantized models]
          ↓
Gets 10× better inference performance on real hardware
```

---

### The Three Levels of Understanding

```
tiny-gpu = pedagogical "bones"
  Makes stalls visible by omitting the muscle.

Snapdragon NPU = "edge muscle"
  Fused compute + shared memory + tiling + INT4 + tight power delivery.

B200 = "datacenter muscle"
  Massive HBM bandwidth + FP4 + NVLink scale-out + TMEM + decompression.
```

### The Correct Common Thread

**Inference is a data-movement problem wearing a math costume.**

Your analysis is directionally right when it treats caching/coalescing/tiling/fusion/QoS as the decisive factors. The remaining discipline is to keep every numeric claim tied either to:

**(a)** an explicit platform spec, or

**(b)** a Roofline/energy bound that can't be argued with.

---

## Chapter 58.5: Complete Module-by-Module Mapping Reference

### Full tiny-gpu → Snapdragon 8 Elite Architecture Map

This table maps every file and every major component in tiny-gpu to its Snapdragon 8 Elite (Adreno 830) equivalent:

| tiny-gpu File | Lines | Key Components | Snapdragon 8 Elite Equivalent | Adreno 830 Lines (Estimated) | Key Differences |
|---|---|---|---|---|---|
| **gpu.sv** | 50 | Top-level instantiation, NUM_CORES=2, parameters | SoC top-level + Adreno 830 instantiation | ~100K | 12 CUs instead of 2 cores; hierarchical slices |
| **dcr.sv** | 20 | Device Control Register, thread_count storage | Command Processor, 1000+ control registers | ~10K | Unified address space instead of simple register |
| **dispatch.sv** | 30 | Simple dispatcher, sequential workgroup launch | Workgroup Scheduler with 40 resident waves per CU | ~20K | Per-CU scheduling instead of global dispatch |
| **core.sv** | 80 | Core FSM (IDLE→EXECUTE→WAIT→UPDATE) | Compute Unit pipeline (fetch→decode→execute) | ~15K | Wave-parallel execution instead of sequential |
| **scheduler.sv** | 120 | Thread-level scheduler, memory stall FSM | Wave Scheduler (dynamic wave selection, branch divergence) | ~25K | 40 waves in-flight instead of 4 threads sequential |
| **fetcher.sv** | 60 | Program counter, instruction fetch from memory | Per-wave PC + instruction cache (16-32KB) | ~10K | I-cache hit → 1 cycle vs always going to memory |
| **decoder.sv** | 50 | Instruction decoder (11 ops) | Instruction decoder (1000+ ops with extensions) | ~8K | Full ISA vs minimal educational ISA |
| **registers.sv** | 40 | Register file (128 bytes total, 4 threads) | Register file (~256 KB per CU, 64 threads) | ~15K | 2000× larger, enables wave residency |
| **alu.sv** | 60 | ALU (ADD/SUB/MUL/DIV, 8-bit) | Full IEEE 754 FP32/FP16 pipelines, transcendentals | ~12K | 192× more ALUs, modern precision |
| **pc.sv** | 30 | Program counter, branching | Per-wave PC + divergence stack | ~8K | Stack-based divergence instead of sequential |
| **lsu.sv** | 100 | Load/Store Unit, simple FSM | LSU + coalescing unit + L1 cache | ~20K | Coalescing 8→1 transactions, caching |
| **controller.sv** | 120 | Memory controller, arbitration FSM | Hierarchical memory (L1/L2/SLC) + QoS arbiter | ~30K | Multi-level caching vs single path |
| **Unused features** | — | Branch divergence (marked TODO) | Full branch divergence with call stacks | — | Not implemented in tiny-gpu |
| **TOTAL** | ~850 | 12 files, ~11 instructions | Adreno 830 GPU | ~200K lines equivalent | 200,000÷850 = 235× larger |

### Specific Parameter Mapping

```
tiny-gpu Parameters                Snapdragon 8 Elite Adreno 830
───────────────────────────────    ────────────────────────────────
NUM_CORES = 2                  →    3 slices × 4 CU = 12 Compute Units
THREADS_PER_BLOCK = 4          →    Waves of 64 fibers (16× larger)
DATA_MEM_SIZE = 256 bytes      →    12 GB LPDDR5X (47 million× larger)
DATA_MEM_DATA_WIDTH = 8 bits   →    32 bits (FP32) or 16 bits (FP16)

PROGRAM_MEM_SIZE = 32 bytes    →    Per-CU I-cache: 16-32 KB
PROGRAM_MEM_ADDR_WIDTH = 5     →    Per-CU 32-bit PC, divergence stacks

NUM_ALU = 8 total              →    1,536 FP32 ALUs total (192×)
ALU_LATENCY = 1 cycle          →    3-4 cycle pipelined (FP32)

CACHE_PRESENT = false          →    L1 (32KB) + L2 (4MB) + SLC (8MB)
COALESCING = false             →    Hardware coalescing 4-64 reqs → 1

MAX_THREADS_IN_FLIGHT = 4      →    40 waves × 64 fibers = 2,560
REGISTER_FILE = 128 bytes      →    ~256 KB (2,000× larger)
```

### Execution Timeline Comparison: Matrix Add, 8 Elements

**tiny-gpu (sequential, no caching, no coalescing):**

```
Cycle 1-4:   EXECUTE [LDR A[i]]
             All 8 threads send separate requests
             × 4-5 cycles each = 32-40 cycles total (sequential)

Cycle 41-44: EXECUTE [LDR B[i]]
             × 4-5 cycles each = 32-40 cycles

Cycle 81-84: EXECUTE [ADD]
             Single instruction, 1 cycle

Cycle 85-88: EXECUTE [STR C[i]]
             × 4-5 cycles each = 32-40 cycles

─────────────────────────────────────────────────
TOTAL: ~120-130 cycles for 8-element matrix add
       ~15-16 cycles per element
       2.7 FLOPS / 8 operands × 120 cycles = ~0.022 FLOPS/byte bandwidth
```

**Adreno 830 (wave scheduled, coalesced, cached):**

```
Cycle 1-4:   Wave 0 issues coalesced load (8 fibers → 1 cache-line req)
             L1 miss, goes to L2 (~20 cycles)

Cycle 2-5:   Wave 1 issues separate coalesced load
             Scheduler switches to Wave 2, 3, etc.

Cycle 6-25:  While Wave 0 waits on L2,
             Waves 2-10 execute their compute phases

Cycle 26:    Wave 0 data returns from L2 to L1

Cycle 27:    Wave 0 issues second load (B[i])
             L1 hit (same cache line) → ~4 cycles

Cycle 31:    Wave 0 issues ADD (coalesced across 64 fibers)
             Pipelined, returns in 3 cycles

Cycle 34:    Wave 0 issues coalesced store (C[i] → L1)
             Write-back cache, returns immediately

Cycle 35:    Wave 0 complete. Scheduler issues next workload.

─────────────────────────────────────────────────
TOTAL: ~35 cycles visible per wave
       ~4-5 cycles per element (8× faster)
       Effective FLOPS with 10 waves in-flight = 30-60× higher throughput
```

---

## Chapter 59: Advanced Learning Path

### Prerequisite Understanding Checklist

Before diving into professional GPU optimization, verify your understanding of:

**Tier 1: Computational Thinking (this guide covers)**
- [ ] SIMD and lockstep execution model
- [ ] Memory hierarchy and latency tiers
- [ ] The Roofline Model (compute vs bandwidth)
- [ ] Branch divergence and warp scheduling
- [ ] Cache coalescing and memory transactions

**Tier 2: Hardware Design (supplementary reading)**
- [ ] Verilog/SystemVerilog syntax (see `references/` in tiny-gpu repo)
- [ ] Digital logic design (gates, flip-flops, FSMs)
- [ ] Timing closure and critical path analysis
- [ ] Power delivery networks (PDN) for high-current designs

**Tier 3: Manufacturing (advanced)**
- [ ] Physical design flow (synthesis → placement → routing)
- [ ] GDS and process design kits (PDK)
- [ ] Photolithography and transistor physics
- [ ] Defect mechanisms (electromigration, TDDB, soft errors)

### Recommended Reading Order

1. **This guide** (Foundations + tiny-gpu + Snapdragon mapping)
2. **tiny-gpu GitHub repository** (actual source code, GDS files, Tiny Tapeout)
3. **Snapdragon 8 Elite specs** (Qualcomm product briefs, teardowns by Chips and Cheese)
4. **NVIDIA CUDA Programming Guide** (professional GPU programming patterns)
5. **Roofline Model papers** (academic analysis of memory-bounded workloads)
6. **OpenGL ES / Vulkan specs** (graphics API mapping to hardware)
7. **Tensor microkernels** (how HTP and TensorRT compile high-level ops)

### Hands-On Exercises

**Exercise 1: Simulate tiny-gpu with Custom Kernel**
- Modify one instruction in alu.sv (e.g., add SQUARE = rs * rs)
- Write a kernel that computes C[i] = A[i]² + B[i]²
- Run the simulator and trace cycles
- Compare cycle count to matrix add

**Exercise 2: Memory Profiling on Real Hardware**
- Install TensorFlow Lite on Android phone with Snapdragon
- Run inference with built-in profiler (GPU vs HTP)
- Observe:
  - Kernel execution time
  - Memory bandwidth used
  - Power draw (via adb shell)
- Compare to Roofline prediction

**Exercise 3: Compiler Optimization**
- Take a quantized model (INT4)
- Disable operator fusion → measure latency
- Re-enable fusion → measure latency
- Compute speedup; tie to roofline model

**Exercise 4: Branch Divergence Analysis**
- Write a kernel with heavy branching (if/else per thread)
- Run on Adreno with profiler
- Observe divergence-induced serialization
- Rewrite to be branch-free; compare

**Exercise 5: Cache Modeling**
- Implement a simplified L1 cache simulator (Least-Recently-Used eviction)
- Run memory access traces from inference
- Compute hit rates vs latency curves
- Tie to roofline model prediction

---

## Chapter 60: Future Directions and Open Questions

### Beyond tiny-gpu: Scaling Challenges

**As designs grow from 100K transistors (tiny-gpu) to 20B transistors (Adreno):**

1. **Power delivery**: Routing power from pads to billions of transistors without voltage drop
2. **Clock distribution**: Keeping 1.2 GHz clock synchronized across 12 km of wiring (relative to transistor size)
3. **Verification**: Testing 20B transistors with limited simulation budget
4. **Yield**: Manufacturing defects become statistically inevitable; add redundancy
5. **Heterogeneity**: mixing HTP + GPU + NPU on same die creates new scheduling problems

### Open Research Areas

1. **Neural Architecture for inference**: Can we auto-generate tiny specialized GPUs for specific models?
2. **Adaptive precision**: Switch FP32 ↔ INT4 dynamically based on layer characteristics?
3. **Dynamic DVFS**: Predict latency needs before kernel launch and set frequency accordingly?
4. **Memory compression**: Decompress weights on-the-fly instead of storing full precision?

### Industry Trends Affecting GPU Design

| Trend | tiny-gpu Era | Snapdragon 8 Elite | Next Generation (2026+) |
|-------|---|---|---|
| **Power wall** | ~5W (phone) | ~10W sustained | < 5W (battery priority) |
| **Compute/memory gap** | 1 ALU : 1 memory channel | 192 ALUs : 4 memory channels | Shrinking (in-memory compute?) |
| **Quantization** | Not implemented | INT4 standard | Mixed INT2/INT4 with dynamic scaling |
| **Latency (ms)** | N/A (simulation) | ~50-100 ms LLM token | < 10 ms (real-time interaction) |
| **Specialization** | General-purpose | HTP (tensor) + GPU (general) | Tile-based with fusion engines |

---

## Appendix: Glossary and Symbol Reference

### Key Terms

**Fiber**: A single scalar processing unit within a wave (synonym: thread in Adreno context)

**Wave**: A group of 64 fibers executing in lockstep (synonym: warp in NVIDIA, workitem group in OpenCL)

**Compute Unit (CU)**: A complete mini-GPU with 6 Shader Processors, Wave Scheduler, L1 cache, and shared memory

**Slice**: Hierarchical grouping of 4 CUs with shared L2 cache and power rail

**Coalescing**: Merging multiple memory requests into a single transaction (e.g., addresses 0, 1, 2, 3 → single 64-byte cache-line fetch)

**Roofline**: Theoretical performance upper bound determined by either compute (peak FLOPS) or bandwidth

**Latency hiding**: Scheduling other work (different waves) while one wave waits on memory

**Write-back cache**: Cache policy where writes stay in cache and are flushed to memory asynchronously

**Microtile**: A small region of activations kept in on-chip memory during inference (e.g., 4×4 tile of image for convolution)

### Symbols and Notation

```
A[i]        Array element, 0-indexed
rs, rt      Source registers (rs = register source, rt = register target)
rPr         Register priority (used in scheduler)
cy          Cycle (e.g., "20 cy" = 20 cycles)
μs          Microsecond (10⁻⁶ seconds)
ns          Nanosecond (10⁻⁹ seconds)
TB/s        Terabytes per second (10¹² bytes/sec)
FLOPS       Floating-point operations per second
TFLOPS      Tera-FLOPS (10¹² FLOPS)
```

---

*Document generated from comprehensive analysis of the [tiny-gpu](https://github.com/AmaadMarian/tiny-gpu) project.*
*Snapdragon specifications sourced from Qualcomm product briefs, independent analysis (NotebookCheck, Tom's Hardware, Chips and Cheese), and Samsung official specs.*
*NVIDIA B200 specifications from NVIDIA DGX B200 page, Blackwell architecture announcements, and academic microbenchmark work.*
*Estimated values marked as such. Proprietary microarchitecture details noted as unverifiable from public sources.*

---

## Document Integrity Verification

This guide contains **93 chapters** across 10 parts (integrated):

| Part | Chapters | Topics |
|------|----------|--------|
| I: Foundations | 0-5 | Hardware physics, transistors, Verilog, manufacturing |
| II: tiny-gpu | 6-19 | All 12 modules + ISA |
| II.5: Code Walkthrough | 19.5-19.16 | Line-by-line code analysis + hardware mapping |
| III: Execution | 20-21 | Kernel launch + cycle trace |
| IV: Problems | 22-26 | Five architectural bottlenecks |
| V: Memory | 27-30 | Banking, stride patterns, arbitration, handshake |
| VI: Snapdragon Mapping | 31-41.5 | Complete module mapping + software stack |
| VII: Inference | 42-50.9 | Side-by-side execution traces |
| VIII: Bandwidth Wall | 51-53.11 | Applied to real hardware + LLM inference + workload analysis |
| IX: NVIDIA B200 | 54-55 | Alternative platform comparison |
| X: Reference | 56-60 | Complete tables, advanced learning, future directions |

**Total lines of content**: 7,200+ (integrated code walkthrough + comprehensive analysis)

**Integrated in this revision:**
- **12 New Code Walkthrough Chapters (19.5-19.16)**:
  - Line-by-line SystemVerilog explanations with concrete examples
  - Execution timelines showing cycle-by-cycle behavior
  - Direct mappings to Snapdragon and B200 equivalent operations
  - Memory access patterns and coalescing analysis
  - Energy breakdown and optimization opportunities
  - Real kernel execution traces with performance comparisons

- **7 First-Principles Analysis Chapters (53.5-53.11)**:
  - Prefill vs Decode workload characterization
  - Quantization effects on bandwidth
  - Systolic vs Dataflow architectural paradigms
  - Shared memory and micro-tiling techniques
  - Prefetching and DMA overlap mechanisms
  - Batch size constraints and register file physics
  - Thermal throttling and sustained vs burst performance
