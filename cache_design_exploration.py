"""
Noah Dumas
CS 4200 Semester Project - Cache Design Exploration
05/03/2026

This program compares different cache organizations:
1. Direct-mapped cache
2. 2-way set associative cache
3. 4-way set associative cache

The simulator supports:
- Tag, index, and block offset addressing
- Valid and dirty bits
- LRU replacement
- Write-back policy
- Write-allocate policy
- Hit and miss detection
- Evictions
- Dirty line writebacks
- Experiment logs and summary results

Run:
    python cache_design_exploration.py

Trace file format:
    L 0x100
    S 0x104
    L 0x108

L = load
S = store
"""

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import sys


@dataclass
class CacheLine:
    valid: bool = False
    dirty: bool = False
    tag: int = 0
    last_used: int = 0


@dataclass
class CacheStats:
    accesses: int = 0
    loads: int = 0
    stores: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    dirty_writebacks: int = 0

    def hit_rate(self):
        if self.accesses == 0:
            return 0.0
        return self.hits / self.accesses

    def miss_rate(self):
        if self.accesses == 0:
            return 0.0
        return self.misses / self.accesses


class CacheSimulator:
    def __init__(self, name, cache_size_bytes, block_size_bytes, associativity):
        self.name = name
        self.cache_size_bytes = cache_size_bytes
        self.block_size_bytes = block_size_bytes
        self.associativity = associativity

        self._check_config()

        self.num_lines = self.cache_size_bytes // self.block_size_bytes
        self.num_sets = self.num_lines // self.associativity

        self.offset_bits = int(math.log2(self.block_size_bytes))
        self.index_bits = int(math.log2(self.num_sets)) if self.num_sets > 1 else 0

        self.sets = []
        for _ in range(self.num_sets):
            cache_set = []
            for _ in range(self.associativity):
                cache_set.append(CacheLine())
            self.sets.append(cache_set)

        self.stats = CacheStats()
        self.clock = 0
        self.access_log = []

    def _is_power_of_two(self, value):
        return value > 0 and (value & (value - 1)) == 0

    def _check_config(self):
        if not self._is_power_of_two(self.cache_size_bytes):
            raise ValueError("Cache size must be a power of 2.")

        if not self._is_power_of_two(self.block_size_bytes):
            raise ValueError("Block size must be a power of 2.")

        if self.cache_size_bytes % self.block_size_bytes != 0:
            raise ValueError("Cache size must be divisible by block size.")

        num_lines = self.cache_size_bytes // self.block_size_bytes

        if associativity_invalid(self.associativity):
            raise ValueError("Associativity must be a positive power of 2.")

        if num_lines % self.associativity != 0:
            raise ValueError("Number of cache lines must be divisible by associativity.")

    def decode_address(self, address):
        block_number = address // self.block_size_bytes
        block_offset = address % self.block_size_bytes
        set_index = block_number % self.num_sets
        tag = block_number // self.num_sets

        return tag, set_index, block_offset, block_number

    def access(self, operation, address, trace_name):
        operation = normalize_operation(operation)

        self.clock += 1
        self.stats.accesses += 1

        if operation == "L":
            self.stats.loads += 1
        elif operation == "S":
            self.stats.stores += 1

        tag, set_index, block_offset, block_number = self.decode_address(address)
        cache_set = self.sets[set_index]

        hit_line = None
        hit_way = -1

        for way, line in enumerate(cache_set):
            if line.valid and line.tag == tag:
                hit_line = line
                hit_way = way
                break

        if hit_line is not None:
            self.stats.hits += 1
            hit_line.last_used = self.clock

            if operation == "S":
                hit_line.dirty = True

            self._log_access(
                trace_name=trace_name,
                operation=operation,
                address=address,
                tag=tag,
                set_index=set_index,
                block_offset=block_offset,
                result="HIT",
                way=hit_way,
                evicted=False,
                dirty_writeback=False,
                victim_tag=None,
                victim_block_address=None
            )
            return "HIT"

        self.stats.misses += 1

        victim_way = self._choose_victim_way(cache_set)
        victim_line = cache_set[victim_way]

        evicted = False
        dirty_writeback = False
        victim_tag = None
        victim_block_address = None

        if victim_line.valid:
            evicted = True
            self.stats.evictions += 1
            victim_tag = victim_line.tag
            victim_block_number = (victim_line.tag * self.num_sets) + set_index
            victim_block_address = victim_block_number * self.block_size_bytes

            if victim_line.dirty:
                dirty_writeback = True
                self.stats.dirty_writebacks += 1

        victim_line.valid = True
        victim_line.dirty = operation == "S"
        victim_line.tag = tag
        victim_line.last_used = self.clock

        self._log_access(
            trace_name=trace_name,
            operation=operation,
            address=address,
            tag=tag,
            set_index=set_index,
            block_offset=block_offset,
            result="MISS",
            way=victim_way,
            evicted=evicted,
            dirty_writeback=dirty_writeback,
            victim_tag=victim_tag,
            victim_block_address=victim_block_address
        )

        return "MISS"

    def _choose_victim_way(self, cache_set):
        for way, line in enumerate(cache_set):
            if not line.valid:
                return way

        lru_way = 0
        lru_time = cache_set[0].last_used

        for way, line in enumerate(cache_set):
            if line.last_used < lru_time:
                lru_time = line.last_used
                lru_way = way

        return lru_way

    def _log_access(
        self,
        trace_name,
        operation,
        address,
        tag,
        set_index,
        block_offset,
        result,
        way,
        evicted,
        dirty_writeback,
        victim_tag,
        victim_block_address
    ):
        log_line = (
            f"trace={trace_name} | "
            f"cache={self.name} | "
            f"access={self.stats.accesses:04d} | "
            f"op={operation} | "
            f"addr=0x{address:08X} | "
            f"tag=0x{tag:X} | "
            f"index={set_index} | "
            f"offset={block_offset} | "
            f"way={way} | "
            f"result={result}"
        )

        if evicted:
            log_line += f" | eviction=YES | victim_tag=0x{victim_tag:X}"

            if victim_block_address is not None:
                log_line += f" | victim_block_addr=0x{victim_block_address:08X}"
        else:
            log_line += " | eviction=NO"

        if dirty_writeback:
            log_line += " | dirty_writeback=YES"
        else:
            log_line += " | dirty_writeback=NO"

        self.access_log.append(log_line)

    def summary(self, trace_name):
        return {
            "trace": trace_name,
            "cache": self.name,
            "cache_size_bytes": self.cache_size_bytes,
            "block_size_bytes": self.block_size_bytes,
            "associativity": self.associativity,
            "sets": self.num_sets,
            "accesses": self.stats.accesses,
            "loads": self.stats.loads,
            "stores": self.stats.stores,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate(),
            "miss_rate": self.stats.miss_rate(),
            "evictions": self.stats.evictions,
            "dirty_writebacks": self.stats.dirty_writebacks
        }


def associativity_invalid(associativity):
    return associativity <= 0 or (associativity & (associativity - 1)) != 0


def normalize_operation(operation):
    operation = operation.upper().strip()

    if operation in ["L", "LOAD", "R", "READ"]:
        return "L"

    if operation in ["S", "STORE", "W", "WRITE"]:
        return "S"

    raise ValueError(f"Invalid operation: {operation}")


def parse_address(value):
    value = value.strip()

    if value.startswith("0x") or value.startswith("0X"):
        return int(value, 16)

    return int(value)


def read_trace_file(path):
    accesses = []

    with open(path, "r") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            line = line.replace(",", " ")
            parts = line.split()

            if len(parts) < 2:
                raise ValueError(f"Bad trace line {line_number}: {line}")

            first = parts[0]
            second = parts[1]

            if first.upper() in ["L", "S", "LOAD", "STORE", "R", "W", "READ", "WRITE"]:
                operation = normalize_operation(first)
                address = parse_address(second)
            else:
                address = parse_address(first)
                operation = normalize_operation(second)

            accesses.append((operation, address))

    return accesses


def write_trace_file(path, accesses):
    with open(path, "w") as file:
        file.write("# Format: operation address\n")
        file.write("# L = load, S = store\n\n")

        for operation, address in accesses:
            file.write(f"{operation} 0x{address:08X}\n")


def build_sample_traces():
    traces = {}

    sequential = []
    base = 0x00000100

    for i in range(32):
        sequential.append(("L", base + (i * 4)))

    for i in range(0, 32, 4):
        sequential.append(("S", base + (i * 4)))

    traces["sequential_array"] = sequential

    repeated_loop = []
    loop_addresses = [
        0x00000200,
        0x00000204,
        0x00000208,
        0x0000020C
    ]

    for _ in range(8):
        for address in loop_addresses:
            repeated_loop.append(("L", address))
        repeated_loop.append(("S", 0x00000208))

    traces["repeated_loop"] = repeated_loop

    two_block_conflict = []
    conflict_addresses = [
        0x00000000,
        0x00000100
    ]

    for _ in range(10):
        for address in conflict_addresses:
            two_block_conflict.append(("S", address))

    traces["two_block_conflict"] = two_block_conflict

    four_block_conflict = []
    four_conflict_addresses = [
        0x00000000,
        0x00000100,
        0x00000200,
        0x00000300
    ]

    for _ in range(8):
        for address in four_conflict_addresses:
            four_block_conflict.append(("L", address))

    traces["four_block_conflict"] = four_block_conflict

    mixed_workload = [
        ("L", 0x00000400),
        ("L", 0x00000404),
        ("L", 0x00000408),
        ("S", 0x0000040C),
        ("L", 0x00000410),
        ("L", 0x00000414),
        ("S", 0x00000500),
        ("S", 0x00000600),
        ("L", 0x00000400),
        ("L", 0x00000404),
        ("S", 0x00000500),
        ("L", 0x00000600),
        ("S", 0x00000700),
        ("L", 0x00000408),
        ("L", 0x0000040C),
        ("S", 0x00000800),
    ]

    traces["mixed_workload"] = mixed_workload

    return traces


def create_sample_trace_files(trace_dir):
    trace_dir.mkdir(exist_ok=True)

    sample_traces = build_sample_traces()
    trace_files = []

    for name, accesses in sample_traces.items():
        path = trace_dir / f"{name}.trace"
        write_trace_file(path, accesses)
        trace_files.append(path)

    return trace_files


def get_cache_configs():
    return [
        {
            "name": "Direct-Mapped",
            "cache_size_bytes": 256,
            "block_size_bytes": 16,
            "associativity": 1
        },
        {
            "name": "2-Way Set Associative",
            "cache_size_bytes": 256,
            "block_size_bytes": 16,
            "associativity": 2
        },
        {
            "name": "4-Way Set Associative",
            "cache_size_bytes": 256,
            "block_size_bytes": 16,
            "associativity": 4
        }
    ]


def run_single_experiment(trace_name, accesses, config):
    cache = CacheSimulator(
        name=config["name"],
        cache_size_bytes=config["cache_size_bytes"],
        block_size_bytes=config["block_size_bytes"],
        associativity=config["associativity"]
    )

    for operation, address in accesses:
        cache.access(operation, address, trace_name)

    return cache.summary(trace_name), cache.access_log


def write_results_csv(path, results):
    fieldnames = [
        "trace",
        "cache",
        "cache_size_bytes",
        "block_size_bytes",
        "associativity",
        "sets",
        "accesses",
        "loads",
        "stores",
        "hits",
        "misses",
        "hit_rate",
        "miss_rate",
        "evictions",
        "dirty_writebacks"
    ]

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            output_row = row.copy()
            output_row["hit_rate"] = f"{row['hit_rate']:.4f}"
            output_row["miss_rate"] = f"{row['miss_rate']:.4f}"
            writer.writerow(output_row)


def write_results_log(path, results):
    with open(path, "w") as file:
        file.write("Cache Design Exploration Results\n")
        file.write("=" * 120)
        file.write("\n\n")

        header = (
            f"{'Trace':<24}"
            f"{'Cache':<26}"
            f"{'Accesses':>10}"
            f"{'Hits':>8}"
            f"{'Misses':>8}"
            f"{'Hit Rate':>12}"
            f"{'Miss Rate':>12}"
            f"{'Evictions':>12}"
            f"{'Writebacks':>12}"
        )

        file.write(header + "\n")
        file.write("-" * 120 + "\n")

        for row in results:
            line = (
                f"{row['trace']:<24}"
                f"{row['cache']:<26}"
                f"{row['accesses']:>10}"
                f"{row['hits']:>8}"
                f"{row['misses']:>8}"
                f"{row['hit_rate'] * 100:>11.2f}%"
                f"{row['miss_rate'] * 100:>11.2f}%"
                f"{row['evictions']:>12}"
                f"{row['dirty_writebacks']:>12}"
            )

            file.write(line + "\n")


def write_full_trace_log(path, all_logs):
    with open(path, "w") as file:
        file.write("Full Cache Access Trace Log\n")
        file.write("=" * 120)
        file.write("\n\n")

        for line in all_logs:
            file.write(line + "\n")


def print_results(results):
    print()
    print("Cache Design Exploration Results")
    print("=" * 120)

    header = (
        f"{'Trace':<24}"
        f"{'Cache':<26}"
        f"{'Accesses':>10}"
        f"{'Hits':>8}"
        f"{'Misses':>8}"
        f"{'Hit Rate':>12}"
        f"{'Miss Rate':>12}"
        f"{'Evictions':>12}"
        f"{'Writebacks':>12}"
    )

    print(header)
    print("-" * 120)

    for row in results:
        line = (
            f"{row['trace']:<24}"
            f"{row['cache']:<26}"
            f"{row['accesses']:>10}"
            f"{row['hits']:>8}"
            f"{row['misses']:>8}"
            f"{row['hit_rate'] * 100:>11.2f}%"
            f"{row['miss_rate'] * 100:>11.2f}%"
            f"{row['evictions']:>12}"
            f"{row['dirty_writebacks']:>12}"
        )

        print(line)

    print("=" * 120)
    print()


def explain_output_files(output_dir):
    print("Generated files:")
    print(f"  {output_dir / 'trace.log'}")
    print(f"  {output_dir / 'results.log'}")
    print(f"  {output_dir / 'results.csv'}")
    print()


def main():
    base_dir = Path(".")
    trace_dir = base_dir / "traces"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        trace_files = [Path(arg) for arg in sys.argv[1:]]
    else:
        trace_files = create_sample_trace_files(trace_dir)

    configs = get_cache_configs()

    all_results = []
    all_logs = []

    for trace_file in trace_files:
        accesses = read_trace_file(trace_file)
        trace_name = trace_file.stem

        for config in configs:
            summary, logs = run_single_experiment(trace_name, accesses, config)
            all_results.append(summary)
            all_logs.extend(logs)
            all_logs.append("")

    write_full_trace_log(output_dir / "trace.log", all_logs)
    write_results_log(output_dir / "results.log", all_results)
    write_results_csv(output_dir / "results.csv", all_results)

    print_results(all_results)
    explain_output_files(output_dir)


if __name__ == "__main__":
    main()
