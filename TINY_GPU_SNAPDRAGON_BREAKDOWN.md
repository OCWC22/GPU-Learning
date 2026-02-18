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

This isn't an opinion or a design choice. It's physics. The canonical reference is Horowitz's energy table (originally 45nm, but the ratios hold across nodes):

```
Operation                    Energy (approximate)
─────────────────────────    ────────────────────
8-bit integer ADD            ~0.03 pJ
32-bit float MUL             ~3.7 pJ
Read 32 bits from SRAM       ~5 pJ
Read 32 bits from DRAM       ~640 pJ

DRAM read is ~170× more expensive than a 32-bit float multiply.
DRAM read is ~128× more expensive than an SRAM read.
```

This means: **the entire purpose of GPU/NPU architecture is to avoid going to DRAM.** Every cache, every coalescing unit, every tiling strategy, every compression engine exists because of this energy gap. The ALUs are almost free by comparison.

tiny-gpu makes this visible by having **no caches, no coalescing, no tiling** — so every operation pays the full DRAM cost. Real chips spend billions of transistors to avoid that cost.

---

## Chapter 2: The Roofline Model

The **Roofline model** tells you whether a workload is limited by compute or by memory bandwidth.

```
Performance (ops/sec)
        │
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

**LLM decode (generating tokens one at a time) is almost always on the left side — bandwidth-limited.** Each token requires reading most of the model's weights, but each weight participates in only one multiply-accumulate. The operational intensity is roughly:

```
For dense decode at batch=1:
  Operations per token: ~2 × parameter_count (one multiply + one accumulate per weight)
  Bytes read per token: ~parameter_count × bytes_per_weight

  Operational intensity ≈ 2 ops / bytes_per_weight

  At INT4 (0.5 bytes per weight): intensity ≈ 4 ops/byte
  At INT8 (1 byte per weight):    intensity ≈ 2 ops/byte
  At FP16 (2 bytes per weight):   intensity ≈ 1 op/byte
```

These are all **very low** operational intensities. For comparison, a well-tiled matrix-matrix multiply can achieve 100+ ops/byte. LLM decode is firmly in the bandwidth-limited regime on every platform.

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

## Chapter 36: scheduler.sv → Wave Scheduler

This is where tiny-gpu and the Adreno diverge most dramatically.

tiny-gpu stalls the entire core when any thread waits for memory. The Adreno keeps **dozens of waves** resident per CU and switches between them:

```
tiny-gpu:
  Block 0: COMPUTE → MEMORY_WAIT → STALL → STALL → STALL → COMPUTE
  ALU utilization: ~1%

Adreno:
  Wave 0: COMPUTE → MEMORY_WAIT...
  Wave 1: COMPUTE → COMPUTE → MEMORY_WAIT...
  Wave 2: COMPUTE → COMPUTE → COMPUTE...
  Wave 0: ...data arrived → COMPUTE
  ALU utilization: ~80-95%
```

This requires massive register files — every resident wave needs its own state kept alive.

---

## Chapter 37: controller.sv → Memory Hierarchy

tiny-gpu has one path: LSU → controller → external memory. Every access, every time.

The Adreno 830 has a deep hierarchy:

```
Coalescing Unit (per-CU)
  → L1 Data Cache (~16-32 KB, ~4 cycles)
    → Slice L2 Cache (4 MB per slice, ~20-40 cycles)
      → SLC (8 MB shared, ~30-40 cycles)
        → LPDDR5X DRAM (12 GB, ~100-150 cycles)
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

# Part VII: Inference Side-by-Side

## Chapter 42: Setup Phase

### tiny-gpu

```python
program_memory.load(program)    # 13 × 16-bit instructions
data_memory.load(data)          # 16 × 8-bit values
dut.device_control_data.value = 8
dut.start.value = 1
```

### Snapdragon 8 Elite

```
Application → OpenCL/Vulkan API → GPU Driver (msm_kgsl)
  → Allocate GPU memory → DMA copy data → Write shader binary
  → Configure CP registers → Kick Command Processor
```

**Key difference:** tiny-gpu loads data directly into 256-byte flat memory. The Snapdragon loads into 12 GB LPDDR5X through DMA, NoC fabric, SLC, and memory controller.

---

## Chapter 43: Dispatch Phase

### tiny-gpu

```
Block 0 → Core 0: blockIdx=0, threads 0-3
Block 1 → Core 1: blockIdx=1, threads 0-3
```

Both cores start simultaneously.

### Snapdragon 8 Elite

```
8 threads, 1 workgroup → 1 wave on 1 CU in Slice 0
Slice 1: power-gated
Slice 2: power-gated
```

For this tiny workload, the Snapdragon **power-gates 2 entire slices.** All 8 threads run in one wave on one CU — no inter-core contention.

---

## Chapter 44: Computing the Thread Index

### tiny-gpu: ~16 cycles for 2 instructions across 2 cores

Each instruction goes through FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE. Program memory contention means Core 1 is always ~2 cycles behind Core 0.

### Snapdragon: ~6 cycles for 2 instructions, all 8 threads

First instruction has a cache miss (~4 cycles). Second instruction hits I-cache (1 cycle). Pipelining overlaps stages.

---

## Chapter 45: The First Memory Load

`LDR R4, R4` — Load A[i] from memory.

### tiny-gpu

8 requests, 4 channels. Controller serves Core 0 first (priority scan). Core 1 blocked for ~3 cycles.

```
Core 0 ALUs: idle  idle  idle  EXEC
Core 1 ALUs: idle  idle  idle  idle  idle  EXEC
```

### Snapdragon 8 Elite

Coalescing unit: 8 addresses [0-7] → 1 transaction. L1 miss → L2 miss → SLC miss → DRAM (~20 cycles). Wave scheduler would switch to other waves (but only 1 wave here, so CU stalls).

**Key point:** tiny-gpu's memory is simulated by Python and responds instantly. Real DRAM takes ~100+ ns. If tiny-gpu had real DRAM latency, it would be far worse.

---

## Chapter 46: The Second Memory Load — Cache Payoff

`LDR R5, R5` — Load B[i] from memory.

### tiny-gpu

Same cost as first load. No caching. Same starvation pattern. Same waste.

### Snapdragon 8 Elite

Coalescing: 8 addresses [8-15] → 1 transaction. **L1 cache HIT!** The first load's cache line fill brought in addresses 0-63 (typical 64-byte cache line). Data returned in ~4 cycles instead of ~100+.

```
                    First LDR (A[i])    Second LDR (B[i])
tiny-gpu:           ~6-8 cycles         ~6-8 cycles        (no cache, same cost)
Adreno 830:         ~20 cycles          ~4 cycles           (L1 hit, 5× faster)
```

---

## Chapter 47: The Actual Computation

`ADD R6, R4, R5` — C[i] = A[i] + B[i]

### tiny-gpu: ~7-8 cycles

Full pipeline (FETCH through UPDATE). The actual computation is 1 cycle. The other 6-7 are overhead.

### Snapdragon: ~1 cycle

I-cache hit, pipelined. Arithmetic instructions are essentially free.

**Key insight:** The actual computation is trivial on both systems. The entire performance difference comes from the memory system and instruction delivery. This is true of real GPU workloads too.

---

## Chapter 48: The Store

`STR R7, R6` — Write C[i] to memory.

### tiny-gpu: ~6-8 cycles

Store goes all the way to external memory, waits for acknowledgment. Same starvation pattern.

### Snapdragon: ~2 cycles

**Write-back caching:** Write hits L1 cache, dirty bit set, core moves on. The cache controller flushes to DRAM later, asynchronously, overlapped with future computation.

---

## Chapter 49: Kernel Complete

### tiny-gpu

Each core signals `done` to the dispatcher. When `blocks_done == total_blocks`, the test detects `dut.done == 1`.

### Snapdragon

GPU writes fence value to memory. CP triggers interrupt to CPU. Driver wakes waiting thread. Application's `clFinish()` returns.

---

## Chapter 50: Complete Timeline Comparison

### tiny-gpu (~104 cycles)

```
13 instructions × ~7 cycles each ≈ ~91 cycles
+ memory stalls (~15-20 cycles)
+ program memory contention (~10-15 cycles)
≈ ~110-130 total cycles

Useful ALU computation: 8 threads × 1 ADD = 8 ALU-cycles
Total ALU-cycles available: 8 ALUs × ~120 cycles = ~960
ALU UTILIZATION: 8 / 960 ≈ 0.8%
```

### Snapdragon (~41 cycles, single wave)

```
I-cache miss on first fetch: ~4 cycles
12 instructions @ ~1 cycle each (pipelined, cached): ~12 cycles
First load (DRAM miss): ~20 cycles
Second load (L1 hit): ~4 cycles
Store (write-back): ~2 cycles
≈ ~41 cycles

With wave scheduling (real workload): ~20 visible cycles
```

### Summary

```
                              tiny-gpu         Adreno 830        Ratio
Total cycles (this kernel)    ~104             ~41               2.5×
Effective cycles (real load)  ~104             ~20               5×
ALU utilization               ~1%              ~60-95%           60-95×
Memory transactions           24 separate      3 coalesced       8×
Instruction fetches           26 to ext mem    2 to ext mem      13×
```

---

# Part VIII: The Bandwidth Wall and Real Inference

## Chapter 51: The Bandwidth Wall — Applied to Real Hardware

### Snapdragon 8 Elite (LPDDR5X)

```
4 channels × 16-bit × 9600 MT/s = 76.8 GB/s raw (~84.8 GB/s spec)
Order of magnitude: ~0.08 TB/s
```

For a 7B parameter model at INT4:

```
Weight payload: 7 × 10⁹ × 0.5 bytes = 3.5 GB

If streaming all weights from DRAM per token:
  tokens/sec ≤ 84.8 / 3.5 ≈ 24 tokens/sec

This is a HARD CEILING from physics.
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

### Same Physics, Different Constants

```
                        Snapdragon 8 Elite    NVIDIA B200
DRAM bandwidth:         ~0.08 TB/s            ~8 TB/s         (100×)
Power budget:           ~5W                   ~1000W          (200×)
On-package memory:      12 GB LPDDR5X         192 GB HBM3e   (16×)
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
  - LPDDR5X bandwidth ceiling: ~84.8 GB/s
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

# Part IX: NVIDIA B200 — Same Physics, Different Constants

## Chapter 54: B200 Architecture Overview

From NVIDIA's public materials:

- **Multi-die packaging:** Two reticle-limit dies connected by high-bandwidth on-package link
- **208 billion transistors** total
- **FP4 native support** via Transformer Engine (halves bytes vs FP8)
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

## Chapter 57: The Complete Mapping Table

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

## Chapter 58: Learning Roadmap

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

*Document generated from comprehensive analysis of the [tiny-gpu](https://github.com/AmaadMartin/tiny-gpu) project.*
*Snapdragon specifications sourced from Qualcomm product briefs, independent analysis (NotebookCheck, Tom's Hardware, Chips and Cheese), and Samsung official specs.*
*NVIDIA B200 specifications from NVIDIA DGX B200 page, Blackwell architecture announcements, and academic microbenchmark work.*
*Estimated values marked as such. Proprietary microarchitecture details noted as unverifiable from public sources.*