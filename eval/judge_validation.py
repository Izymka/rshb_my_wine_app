"""Проверка вариантов VLM-судьи на живых знакомых кадрах train — без теста.

Зачем. Правила судьи (какие карточки показывать, когда звать, проверять ли выбор текстом,
думать ли модели) нельзя подбирать по `data/test`: тогда 90 % на тесте перестанут что-то
значить. Здесь варианты сравниваются на живых кадрах знакомых вин из `data/train`
(organizer_100, фото из интернета, свои), вина теста исключены целиком.

Две проверки на каждый вариант:

- **естественная** — пайплайн как в сервисе, судья по своему правилу вызова: сколько вызовов,
  сколько верных top-1 исправлено и сколько испорчено. Решающий слой обучен на этих же кадрах,
  поэтому ошибок у него здесь мало, и исправлять почти нечего — зато поломки видны честно;
- **стресс** — на кадрах, где лидер верный и среди кандидатов есть сосед по винодельне, первым
  ставится сильнейший сосед (как будто ошибся решающий слой). Судья должен вернуть верное вино.
  Это мера того, ради чего судья существует, на достаточном числе примеров.

    uv run python eval/judge_validation.py --variants all            # нужны ключи Yandex в окружении
    uv run python eval/judge_validation.py --variants family-verify,family-verify-low --limit 20

Результаты: строка на вариант в `eval/results/judge_validation_runs.jsonl`, вызовы по кадрам —
`eval/results/judge_validation_calls.jsonl` (дописывается, с меткой прогона).
"""

import argparse
import json
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from wine_scanner.catalog import TEST_MANIFEST, is_holdout, load_equivalences, load_manifest
from wine_scanner.decide.guard import GRAPE_ENABLED
from wine_scanner.decide.judge import VlmJudge, describe
from wine_scanner.embed import load_image
from wine_scanner.llm import chat_from_env
from wine_scanner.pipeline import WineScanner

TRAIN_MANIFEST = Path("data/train/manifest.csv")
CROP_CACHE = Path("models/crop_cache")
RUNS = Path("eval/results/judge_validation_runs.jsonl")
CALLS = Path("eval/results/judge_validation_calls.jsonl")
# Дубли каталога по решению человека: выбрать вторую карточку того же вина — не ошибка.
EQUIVALENCES = load_equivalences()


@dataclass(frozen=True)
class Variant:
    options: str
    trigger: str
    verify: bool
    reasoning: str  # reasoning_effort для Yandex; "none" — без рассуждений
    ocr: str = "off"  # строки PaddleOCR судье: off / hint / hint-verify (judge.OCR_MODES)
    check: bool = False  # сверять выбор судьи с его же прочитанным текстом (consistent_pick)


VARIANTS = {
    # Как в прогоне 7e17c62: 8 карточек, правило judge_reason, без проверки.
    "all-reasons": Variant("all", "reasons", False, "none"),
    "family-reasons": Variant("family", "reasons", False, "none"),
    "family-always": Variant("family", "family", False, "none"),
    "family-verify": Variant("family", "family", True, "none"),
    "family-verify-low": Variant("family", "family", True, "low"),
    # Строки PaddleOCR в промпте; смену лидера подтверждает только текст судьи.
    "family-verify-ocr": Variant("family", "family", True, "none", "hint"),
    # То же, но смену лидера может подтвердить и текст OCR.
    "family-verify-ocr-check": Variant("family", "family", True, "none", "hint-verify"),
    # Выбор судьи сверяется с его же текстом: цвет, сорт, сладость, различающие слова.
    "family-verify-check": Variant("family", "family", True, "none", "off", True),
}


@dataclass
class Frame:
    path: str
    true_id: str
    source: str
    crop: object
    candidates: list
    answered: bool
    ocr_lines: list[str]


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "?"


def load_frames(scanner: WineScanner, limit: int | None) -> list[Frame]:
    """Прогнать пайплайн без судьи по живым знакомым кадрам train, вина теста — долой."""
    test = pd.read_csv(TEST_MANIFEST, dtype=str, keep_default_na=False)
    test_wines = set(test.true_slug) - {""}
    queries = [
        q
        for q in load_manifest(TRAIN_MANIFEST)
        if not q.true_id.startswith("unknown:") and q.true_id not in test_wines
    ]
    assert not any(is_holdout(q.path) for q in queries), "в стенд попал кадр теста"
    if limit:
        queries = queries[:limit]
    frames = []
    for query in tqdm(queries, desc="пайплайн"):
        image = load_image(query.path)
        key = str(query.path)
        result = scanner.identify(image, image_key=key, trace=True)
        frames.append(
            Frame(
                path=key,
                true_id=query.true_id,
                source=query.source,
                crop=scanner._crop(image, key),
                candidates=result.candidates,
                answered=result.answered,
                ocr_lines=list(result.trace["ocr_lines"]),
            )
        )
    return frames


def stressed(frame: Frame, family_of: dict, max_options: int) -> list | None:
    """Кандидаты с соседом по винодельне на первом месте вместо верного лидера."""
    candidates = frame.candidates
    if not candidates or not EQUIVALENCES.same(frame.true_id, candidates[0].item_id):
        return None
    family = family_of.get(frame.true_id)
    sibling = next(
        (
            c
            for c in candidates[1:max_options]
            if family
            and family_of.get(c.item_id) == family
            and not EQUIVALENCES.same(frame.true_id, c.item_id)
        ),
        None,
    )
    if sibling is None:
        return None
    return [sibling] + [c for c in candidates if c is not sibling]


def run_variant(
    name: str, variant: Variant, frames, family_of, threshold, llm, timeout, text_index
):
    llm.reasoning_effort = variant.reasoning
    judge = VlmJudge(
        llm=llm,
        timeout=timeout,
        options=variant.options,
        trigger=variant.trigger,
        verify=variant.verify,
        ocr=variant.ocr,
        check=variant.check,
    )
    calls, natural, stress = [], [], []
    for frame in tqdm(frames, desc=name):
        # Естественная проверка: копии кандидатов — судья меняет вероятности на месте.
        before = frame.candidates[0].item_id if frame.candidates else ""
        candidates = [_copy(c) for c in frame.candidates]
        started = time.perf_counter()
        after_list, _, report = judge.consult(
            frame.crop,
            candidates,
            frame.answered,
            threshold,
            family_of,
            frame.ocr_lines,
            text_index,
        )
        elapsed = time.perf_counter() - started
        after = after_list[0].item_id if after_list else ""
        natural.append(
            (
                EQUIVALENCES.same(frame.true_id, before),
                EQUIVALENCES.same(frame.true_id, after),
                report is not None,
            )
        )
        if report is not None:
            calls.append(_call(name, "natural", frame, report, elapsed, before, after))

        # Стресс: правило вызова не спрашиваем — спорный случай создан руками.
        swapped = stressed(frame, family_of, judge.max_options)
        if swapped is None:
            continue
        shown = (
            swapped[: judge.max_options]
            if variant.options == "all"
            else _family(swapped, family_of, judge)
        )
        started = time.perf_counter()
        verdict = judge.judge(frame.crop, [describe(c) for c in shown], frame.ocr_lines)
        elapsed = time.perf_counter() - started
        leader = swapped[0]
        chosen, outcome = judge.settle(verdict, shown, leader, frame.ocr_lines, text_index)
        if outcome == "unverified":
            chosen = None
        final = chosen.item_id if chosen is not None else leader.item_id
        stress.append((EQUIVALENCES.same(frame.true_id, final), bool(verdict.error)))
        calls.append(
            _call(
                name,
                "stress",
                frame,
                {
                    "choice": verdict.choice,
                    "read_text": verdict.read_text,
                    "error": verdict.error,
                    "applied": outcome,
                },
                elapsed,
                leader.item_id,
                final,
            )
        )
    return judge, calls, natural, stress


def _copy(candidate):
    from dataclasses import replace

    return replace(candidate, payload=dict(candidate.payload), features=dict(candidate.features))


def _family(candidates, family_of, judge):
    from wine_scanner.decide.judge import family_options

    return family_options(candidates, family_of, judge.max_options, judge.max_family)


def _call(variant, kind, frame, report, elapsed, before, after):
    return {
        "variant": variant,
        "kind": kind,
        "path": frame.path,
        "source": frame.source,
        "true_id": frame.true_id,
        "before": before,
        "after": after,
        "seconds": round(elapsed, 3),
        **{k: report.get(k) for k in ("reason", "options", "choice", "applied", "error")},
        "read_text": report.get("read_text"),
    }


def summarize(name, variant, judge, calls, natural, stress):
    seconds = [c["seconds"] for c in calls]
    return {
        "variant": name,
        **variant.__dict__,
        "frames": len(natural),
        "natural_calls": sum(called for _, _, called in natural),
        "top1_before": sum(b for b, _, _ in natural),
        "top1_after": sum(a for _, a, _ in natural),
        "fixed": sum(a and not b for b, a, _ in natural),
        "broken": sum(b and not a for b, a, _ in natural),
        "stress_n": len(stress),
        "stress_recovered": sum(ok for ok, _ in stress),
        "errors": judge.errors,
        "unverified": sum(c.get("applied") == "unverified" for c in calls),
        "consistent": sum(c.get("applied") == "consistent" for c in calls),
        "call_p50_s": round(statistics.median(seconds), 2) if seconds else None,
        "call_p95_s": round(sorted(seconds)[int(0.95 * (len(seconds) - 1))], 2) if seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("models/index_platform_sq_v3"))
    parser.add_argument("--decider", type=Path, default=Path("models/decider_platform_sq_v3_slug"))
    parser.add_argument("--variants", default="all", help=f"через запятую из {sorted(VARIANTS)}")
    parser.add_argument("--timeout", type=float, default=30.0, help="таймаут вызова для замера")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    names = sorted(VARIANTS) if args.variants == "all" else args.variants.split(",")
    unknown = set(names) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"неизвестные варианты: {sorted(unknown)}")

    scanner = WineScanner(
        index_dir=args.index, decider_dir=args.decider, crop_cache=CROP_CACHE, judge=None
    )
    assert scanner.judge is None, "судья пайплайна должен быть выключен: WINE_VLM=0"
    frames = load_frames(scanner, args.limit)
    family_of = scanner.text_index.family_of
    llm = chat_from_env("WINE_VLM", timeout=args.timeout)
    print(f"кадров: {len(frames)}, верный лидер до судьи: "
          f"{sum(EQUIVALENCES.same(f.true_id, f.candidates[0].item_id) for f in frames if f.candidates)}")

    stamp = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git": git_head(), "tag": args.tag,
             "index": str(args.index), "decider": str(args.decider), "frames": len(frames),
             "grape_rule": GRAPE_ENABLED}
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        judge, calls, natural, stress = run_variant(
            name,
            VARIANTS[name],
            frames,
            family_of,
            scanner.threshold,
            llm,
            args.timeout,
            scanner.text_index,
        )
        row = summarize(name, VARIANTS[name], judge, calls, natural, stress)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
        # Пишем сразу после варианта: медленный вариант можно оборвать, не теряя готовых.
        with RUNS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**stamp, **row}, ensure_ascii=False) + "\n")
        with CALLS.open("a", encoding="utf-8") as fh:
            for call in calls:
                fh.write(json.dumps({**stamp, **call}, ensure_ascii=False) + "\n")

    table = pd.DataFrame(rows).set_index("variant")
    print(table[["natural_calls", "top1_before", "top1_after", "fixed", "broken", "stress_n",
                 "stress_recovered", "unverified", "consistent", "errors", "call_p50_s", "call_p95_s"]])


if __name__ == "__main__":
    main()
