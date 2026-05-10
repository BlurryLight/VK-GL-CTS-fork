#!/usr/bin/env python3
#
# Run Raspberry Pi 5 v3dv Vulkan CTS with Mesa CI-style expectations.
#
# This script is intentionally host-driven. It prepares the same core inputs
# Mesa CI uses for v3dv-rpi5-vk:
#
#   - vk-main.txt generated from external/vulkancts/mustpass/main/vk-default.txt
#   - Mesa's .gitlab-ci/all-skips.txt
#   - Mesa's src/broadcom/ci/broadcom-rpi5-{skips,fails,flakes}.txt
#
# It then syncs those files to an existing rpi5_v3dv_cts package and runs the
# package's run-pi5-v3dv.sh wrapper over ssh. It does not require deqp-runner on
# the Pi; the result classifier below applies Mesa's skip/fail/flake lists.
#
# Typical smoke run:
#
#   scripts/run_pi5_v3dv_ci_cts.py --pi pi5 --smoke 20
#
# Mesa pre-merge sampling equivalent:
#
#   scripts/run_pi5_v3dv_ci_cts.py --pi pi5 --mode ci-sample
#
# Full v3dv-rpi5-vk-full equivalent, split like Mesa CI's 4 shards:
#
#   scripts/run_pi5_v3dv_ci_cts.py --pi pi5 --mode ci-full
#
# If your Mesa v3dv prefix on the Pi is not ~/rpi5_v3dv_debug:
#
#   scripts/run_pi5_v3dv_ci_cts.py --pi pi5 --mesa-prefix ~/rpi5_v3dv --mode ci-full
#
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_DIR = REPO_ROOT / "build-rpi5-v3dv-debug" / "rpi5_v3dv_cts_debug"
DEFAULT_MESA_DIR = Path("/home/panda/repo/mesa")

PASS_STATUSES = {
    "Pass",
    "NotSupported",
    "CompatibilityWarning",
    "QualityWarning",
}


@dataclass(frozen=True)
class Expectation:
    pattern: str
    source: str
    regex: re.Pattern[str] | None


@dataclass(frozen=True)
class Expectations:
    exact: dict[str, Expectation]
    prefix: list[tuple[str, Expectation]]
    prefix_values: tuple[str, ...]
    regex: list[Expectation]


@dataclass
class CaseResult:
    name: str
    status: str
    detail: str
    classification: str


def run(command: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(str(part)) for part in command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def read_patterns(path: Path, *, vk_only: bool = False) -> list[str]:
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.split(",", 1)[0].strip()
        if vk_only and "dEQP-VK" not in line:
            continue
        patterns.append(line)
    return patterns


def has_regex_syntax(pattern: str) -> bool:
    # Mesa's lists are regexes, but most dEQP case lines are intended as exact
    # case names. Treat dots as literal case separators for speed.
    return any(char in pattern for char in "*[]()+?{}\\^$|")


def simple_prefix(pattern: str) -> str | None:
    if not pattern.endswith("*"):
        return None
    prefix = pattern[:-1]
    if any(char in prefix for char in "*[]()+?{}\\^$|"):
        return None
    if prefix.endswith("."):
        prefix = prefix[:-1]
    return prefix


def compile_expectations(paths: list[Path], *, vk_only: bool = True) -> Expectations:
    exact: dict[str, Expectation] = {}
    prefixes: list[tuple[str, Expectation]] = []
    regex_expectations: list[Expectation] = []
    for path in paths:
        if not path.exists():
            continue
        for pattern in read_patterns(path, vk_only=vk_only):
            prefix = simple_prefix(pattern)
            if prefix is not None:
                expectation = Expectation(pattern=pattern, source=path.name, regex=None)
                prefixes.append((prefix, expectation))
                continue
            if not has_regex_syntax(pattern):
                exact[pattern] = Expectation(pattern=pattern, source=path.name, regex=None)
                continue
            try:
                regex = re.compile(pattern)
            except re.error as err:
                raise SystemExit(f"Invalid regex in {path}: {pattern!r}: {err}") from err
            regex_expectations.append(Expectation(pattern=pattern, source=path.name, regex=regex))
    prefixes.sort(key=lambda item: len(item[0]), reverse=True)
    return Expectations(exact=exact, prefix=prefixes, prefix_values=tuple(prefix for prefix, _ in prefixes), regex=regex_expectations)


def first_match(case: str, expectations: Expectations) -> Expectation | None:
    exact_match = expectations.exact.get(case)
    if exact_match:
        return exact_match
    if expectations.prefix_values and case.startswith(expectations.prefix_values):
        for prefix, expectation in expectations.prefix:
            if case.startswith(prefix):
                return expectation
    for expectation in expectations.regex:
        if expectation.regex and expectation.regex.search(case):
            return expectation
    return None


def read_mustpass_cases(mustpass_root: Path) -> list[str]:
    default_list = mustpass_root / "vk-default.txt"
    cases: list[str] = []

    for raw in default_list.read_text(encoding="utf-8").splitlines():
        rel = raw.strip()
        if not rel or rel.startswith("#"):
            continue
        case_file = mustpass_root / rel
        if not case_file.exists():
            raise SystemExit(f"Mustpass file listed by {default_list} is missing: {case_file}")
        for case_raw in case_file.read_text(encoding="utf-8").splitlines():
            case = case_raw.strip()
            if case and not case.startswith("#"):
                cases.append(case)

    return cases


def shard_cases(cases: list[str], fraction_start: int, fraction: int) -> list[str]:
    if fraction < 1:
        raise SystemExit("--fraction must be >= 1")
    if fraction_start < 1 or fraction_start > fraction:
        raise SystemExit("--fraction-start must be between 1 and --fraction")
    return [case for index, case in enumerate(cases) if index % fraction == fraction_start - 1]


def split_evenly(cases: list[str], shards: int) -> list[list[str]]:
    if shards < 1:
        raise SystemExit("shards must be >= 1")
    return [shard_cases(cases, start, shards) for start in range(1, shards + 1)]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def prepare_ci_inputs(args: argparse.Namespace) -> dict[str, object]:
    package_dir = args.package_dir.resolve()
    mesa_dir = args.mesa_dir.resolve()
    work_dir = package_dir / "ci-v3dv"
    mustpass_root = REPO_ROOT / "external" / "vulkancts" / "mustpass" / "main"

    all_cases = read_mustpass_cases(mustpass_root)

    skips = compile_expectations(
        [
            mesa_dir / ".gitlab-ci" / "all-skips.txt",
            mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-skips.txt",
        ]
    )
    fails = compile_expectations([mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-fails.txt"])
    flakes = compile_expectations([mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-flakes.txt"])

    runnable: list[str] = []
    skipped: list[str] = []
    expected_failures: list[str] = []
    flaky: list[str] = []
    must_pass: list[str] = []

    for case in all_cases:
        if first_match(case, skips):
            skipped.append(case)
            continue
        runnable.append(case)
        fail_match = first_match(case, fails)
        flake_match = first_match(case, flakes)
        if fail_match:
            expected_failures.append(case)
        elif flake_match:
            flaky.append(case)
        else:
            must_pass.append(case)

    write_lines(work_dir / "vk-main.txt", all_cases)
    write_lines(work_dir / "run-list.txt", runnable)
    write_lines(work_dir / "skipped-expanded.txt", skipped)
    write_lines(work_dir / "expected-failures-expanded.txt", expected_failures)
    write_lines(work_dir / "flakes-expanded.txt", flaky)
    write_lines(work_dir / "must-pass.txt", must_pass)

    skip_sources = [
        mesa_dir / ".gitlab-ci" / "all-skips.txt",
        mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-skips.txt",
    ]
    write_lines(work_dir / "skips.txt", [p for path in skip_sources for p in read_patterns(path, vk_only=True)])
    write_lines(work_dir / "fails.txt", read_patterns(mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-fails.txt", vk_only=True))
    write_lines(work_dir / "flakes.txt", read_patterns(mesa_dir / "src" / "broadcom" / "ci" / "broadcom-rpi5-flakes.txt", vk_only=True))

    return {
        "package_dir": package_dir,
        "work_dir": work_dir,
        "all_cases": all_cases,
        "runnable": runnable,
        "skipped": skipped,
        "expected_failures": expected_failures,
        "flaky": flaky,
        "must_pass": must_pass,
        "fails": fails,
        "flakes": flakes,
    }


def choose_shards(args: argparse.Namespace, runnable: list[str]) -> list[tuple[str, list[str]]]:
    if args.smoke:
        return [("smoke", runnable[: args.smoke])]

    if args.mode == "ci-sample":
        total_fractions = 12
        starts = args.fraction_start or [1, 2]
        return [(f"ci-sample-{start}-of-{total_fractions}", shard_cases(runnable, start, total_fractions)) for start in starts]

    if args.mode == "ci-full":
        shards = split_evenly(runnable, 4)
        return [(f"ci-full-{index}-of-4", shard) for index, shard in enumerate(shards, start=1)]

    if args.mode == "all":
        return [("all", runnable)]

    raise SystemExit(f"Unhandled mode: {args.mode}")


def sync_to_pi(pi: str, package_dir: Path, remote_dir: str) -> None:
    run(["rsync", "-az", "--delete", f"{package_dir}/", f"{pi}:{remote_dir.rstrip('/')}/"])


def remote_sh(pi: str, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["ssh", pi, f"sh -lc {shlex.quote(command)}"], check=check)


def remote_shell_path(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + shlex.quote(path[2:])
    return shlex.quote(path)


def render_remote_command(remote_dir: str, mesa_prefix: str, list_name: str, qpa_name: str) -> str:
    remote_dir_q = remote_shell_path(remote_dir.rstrip("/"))
    mesa_prefix_q = remote_shell_path(mesa_prefix)
    list_q = shlex.quote(f"ci-v3dv/{list_name}")
    qpa_q = shlex.quote(f"ci-v3dv/results/{qpa_name}")
    return (
        f"cd {remote_dir_q} && "
        "mkdir -p ci-v3dv/results && "
        f"MESA_PREFIX={mesa_prefix_q} ./run-pi5-v3dv.sh "
        f"--deqp-caselist-file={list_q} "
        f"--deqp-log-filename={qpa_q}"
    )


def parse_qpa(path: Path, fails: Expectations, flakes: Expectations) -> list[CaseResult]:
    results: list[CaseResult] = []
    current: str | None = None
    test_re = re.compile(r"Test case '([^']+)'\.\.")
    qpa_begin_re = re.compile(r"#beginTestCaseResult\s+(\S+)")
    qpa_result_re = re.compile(r'<Result\s+StatusCode="([^"]+)">(.*)</Result>')
    result_re = re.compile(r"\b(Pass|Fail|NotSupported|QualityWarning|CompatibilityWarning|ResourceError|InternalError|Crash|Timeout)\b(?: \((.*)\))?")

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        test_match = test_re.search(raw)
        if test_match:
            current = test_match.group(1)
            continue

        qpa_begin_match = qpa_begin_re.search(raw)
        if qpa_begin_match:
            current = qpa_begin_match.group(1)
            continue

        if current is None:
            continue

        stripped = raw.strip()
        qpa_result_match = qpa_result_re.search(stripped)
        if qpa_result_match:
            status = qpa_result_match.group(1)
            detail = qpa_result_match.group(2)
        else:
            result_match = result_re.search(stripped)
            if not result_match:
                continue
            status = result_match.group(1)
            detail = result_match.group(2) or ""

        if status == "NotSupported":
            status = "NotSupported"
        elif status == "ResourceError":
            status = "ResourceError"
        elif status == "InternalError":
            status = "InternalError"

        fail_match = first_match(current, fails)
        flake_match = first_match(current, flakes)
        passed = status in PASS_STATUSES

        if fail_match:
            classification = "expected-fail-pass" if passed else "expected-fail"
        elif flake_match:
            classification = "flake-pass" if passed else "flake-fail"
        elif passed:
            classification = "must-pass"
        else:
            classification = "unexpected-fail"

        results.append(CaseResult(current, status, detail, classification))
        current = None

    return results


def write_results_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["case", "status", "detail", "classification"])
        for result in results:
            writer.writerow([result.name, result.status, result.detail, result.classification])


def summarize(results: list[CaseResult]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for result in results:
        summary[result.classification] = summary.get(result.classification, 0) + 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Pi5 v3dv Vulkan CTS with Mesa CI-style expectations.")
    parser.add_argument("--pi", default="pi5", help="SSH target for the Raspberry Pi 5.")
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR, help="Local staged CTS package dir.")
    parser.add_argument("--remote-dir", default="~/rpi5_v3dv_cts_debug", help="Remote CTS package dir on the Pi.")
    parser.add_argument("--mesa-dir", type=Path, default=DEFAULT_MESA_DIR, help="Mesa checkout containing Broadcom CI files.")
    parser.add_argument("--mesa-prefix", default="~/rpi5_v3dv_debug", help="Mesa v3dv install prefix on the Pi.")
    parser.add_argument("--mode", choices=["ci-sample", "ci-full", "all"], default="ci-sample")
    parser.add_argument("--smoke", type=int, help="Run only the first N runnable cases after CI skip filtering.")
    parser.add_argument("--fraction-start", type=int, action="append", help="For ci-sample, run specific 1-based fractions out of 12.")
    parser.add_argument("--no-sync", action="store_true", help="Do not rsync the package before running.")
    parser.add_argument("--dry-run", action="store_true", help="Only generate local lists and print the remote commands.")
    parser.add_argument("--renderer-check", default="V3D 7.1.7", help="Expected renderer substring. Use empty string to disable.")
    args = parser.parse_args()

    prepared = prepare_ci_inputs(args)
    package_dir = prepared["package_dir"]
    work_dir = prepared["work_dir"]
    runnable = prepared["runnable"]
    fails = prepared["fails"]
    flakes = prepared["flakes"]

    if not (package_dir / "run-pi5-v3dv.sh").exists():
        raise SystemExit(f"Missing run-pi5-v3dv.sh in package dir: {package_dir}")

    shards = choose_shards(args, runnable)
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Prepared Mesa CI-style v3dv CTS lists:\n"
        f"  all vk-main cases: {len(prepared['all_cases'])}\n"
        f"  runnable after skips: {len(prepared['runnable'])}\n"
        f"  skipped: {len(prepared['skipped'])}\n"
        f"  expected failures: {len(prepared['expected_failures'])}\n"
        f"  flakes: {len(prepared['flaky'])}\n"
        f"  must pass: {len(prepared['must_pass'])}\n"
        f"  local work dir: {work_dir}"
    )

    for shard_name, cases in shards:
        list_name = f"{shard_name}.txt"
        write_lines(work_dir / list_name, cases)
        command = render_remote_command(args.remote_dir, args.mesa_prefix, list_name, f"{shard_name}.qpa")
        print(f"\nShard {shard_name}: {len(cases)} cases")
        print(f"  ssh {args.pi} {shlex.quote(command)}")

    if args.dry_run:
        return 0

    if not args.no_sync:
        sync_to_pi(args.pi, package_dir, args.remote_dir)

    if args.renderer_check:
        write_lines(work_dir / "renderer-check.txt", ["dEQP-VK.info.device"])
        if not args.no_sync:
            sync_to_pi(args.pi, package_dir, args.remote_dir)
        remote_sh(
            args.pi,
            render_remote_command(args.remote_dir, args.mesa_prefix, "renderer-check.txt", "renderer-check.qpa"),
            check=False,
        )
        run(["rsync", "-az", f"{args.pi}:{args.remote_dir.rstrip('/')}/ci-v3dv/results/renderer-check.qpa", str(results_dir / "renderer-check.qpa")])
        renderer_qpa = (results_dir / "renderer-check.qpa").read_text(encoding="utf-8", errors="replace")
        if args.renderer_check not in renderer_qpa:
            raise SystemExit(f"Renderer check failed: {args.renderer_check!r} was not found in renderer-check.qpa")

    all_results: list[CaseResult] = []
    failed = False
    for shard_name, _cases in shards:
        command = render_remote_command(args.remote_dir, args.mesa_prefix, f"{shard_name}.txt", f"{shard_name}.qpa")
        completed = remote_sh(args.pi, command, check=False)
        if completed.returncode != 0:
            print(f"deqp-vk returned {completed.returncode} for {shard_name}; parsing QPA anyway.", file=sys.stderr)

        qpa_local = results_dir / f"{shard_name}.qpa"
        run(["rsync", "-az", f"{args.pi}:{args.remote_dir.rstrip('/')}/ci-v3dv/results/{shard_name}.qpa", str(qpa_local)])
        shard_results = parse_qpa(qpa_local, fails, flakes)
        write_results_csv(results_dir / f"{shard_name}.csv", shard_results)
        all_results.extend(shard_results)

        shard_summary = summarize(shard_results)
        print(f"Result summary for {shard_name}: {shard_summary}")
        if shard_summary.get("unexpected-fail", 0):
            failed = True

    write_results_csv(results_dir / "results.csv", all_results)
    final_summary = summarize(all_results)
    print(f"\nFinal summary: {final_summary}")
    print(f"Results written under: {results_dir}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
