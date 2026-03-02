from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import retargeting as rt  # noqa: E402

from optimal_embodiment.eval import robot2human as r2h  # noqa: E402


def _parse_seed_list(seed_list: Optional[str]) -> Optional[List[int]]:
    if seed_list is None:
        return None
    raw = [x.strip() for x in seed_list.split(",")]
    seeds = [int(x) for x in raw if x]
    if not seeds:
        raise ValueError("`--robot-seeds` was provided but ended up empty.")
    return seeds


def _build_seed_plan(seed: int, num_robots: int, seed_list: Optional[str]) -> List[int]:
    parsed = _parse_seed_list(seed_list)
    if parsed is not None:
        return parsed
    if num_robots <= 0:
        raise ValueError("`--num-robots` must be > 0.")
    return [int(seed) + i for i in range(int(num_robots))]


def _fmt(x: float, d: int = 3) -> str:
    try:
        if x != x:
            return "nan"
    except Exception:
        return "nan"
    return f"{x:.{d}f}"


def _robot_dirs_with_scores(
    results: Sequence[Tuple[int, r2h.RobotEval]],
    top_k: int,
    worst_k: int,
) -> List[Tuple[int, r2h.RobotEval]]:
    if not results:
        return []
    ordered = sorted(results, key=lambda x: x[1].overall_score, reverse=True)
    selected: List[Tuple[int, r2h.RobotEval]] = []
    seen: set[str] = set()

    for pair in ordered[: max(0, top_k)]:
        if pair[1].robot_dir not in seen:
            selected.append(pair)
            seen.add(pair[1].robot_dir)
    if worst_k > 0:
        for pair in reversed(ordered[-worst_k:]):
            if pair[1].robot_dir not in seen:
                selected.append(pair)
                seen.add(pair[1].robot_dir)
    return selected


def _save_per_robot_score(robot_dir: Path, seed: int, result: r2h.RobotEval) -> None:
    payload = {
        "seed": seed,
        "result": asdict(result),
    }
    out = robot_dir / "score.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full pipeline: human npz -> random robots -> retargeting -> scoring -> "
            "visualization ordered by score."
        )
    )
    parser.add_argument("--motion-npz", type=str, required=True, help="Human motion .npz (AMASS/SMPLH).")
    parser.add_argument(
        "--body-models-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Directory with body_models/ (SMPLH + DMPLs).",
    )
    parser.add_argument(
        "--template-xml",
        type=str,
        default=str(ROOT / "assets" / "g1_29dof" / "g1_29dof.xml"),
        help="Base XML for randomization.",
    )
    parser.add_argument(
        "--template-ik",
        type=str,
        default=str(
            ROOT
            / "third_party"
            / "gmr"
            / "general_motion_retargeting"
            / "ik_configs"
            / "smplx_to_g1.json"
        ),
        help="Reference IK to generate dynamic IK per robot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs" / "full_pipeline_eval"),
        help="Output folder.",
    )
    parser.add_argument(
        "--motion-stem",
        type=str,
        default=None,
        help="Name of pkl without extension (default: npz stem).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed (if not using --robot-seeds).")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots (if not using --robot-seeds).")
    parser.add_argument(
        "--robot-seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds. Example: 1,7,42,105",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--disable-head-task", action="store_true", default=False)
    parser.add_argument("--disable-head-joints", action="store_true", default=False)
    parser.add_argument("--enable-dtw", action="store_true", default=False, help="Include DTW in score (slower).")
    parser.add_argument("--max-dtw-frames", type=int, default=600)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        default=False,
        help="If robot.xml/IK/PKL already exists for a robot, do not regenerate.",
    )
    parser.add_argument(
        "--visualize-ranked",
        action="store_true",
        default=False,
        help="At the end, play back ranked robots showing original human + robot.",
    )
    parser.add_argument("--visualize-top-k", type=int, default=3, help="Top-K to visualize.")
    parser.add_argument("--visualize-worst-k", type=int, default=1, help="Worst-K to visualize.")
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        default=False,
        help="If set, do not respect real fps in final visualization.",
    )
    parser.add_argument("--record-video", action="store_true", default=False, help="Record mp4 in final visualization.")
    parser.add_argument("--summary-json", type=str, default=None, help="Path of final JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motion_npz = Path(args.motion_npz).resolve()
    body_models_dir = Path(args.body_models_dir).resolve()
    template_xml = Path(args.template_xml).resolve()
    template_ik = Path(args.template_ik).resolve()
    output_dir = Path(args.output_dir).resolve()
    motion_stem = args.motion_stem if args.motion_stem else motion_npz.stem
    summary_json = Path(args.summary_json).resolve() if args.summary_json else output_dir / "summary.json"

    if not motion_npz.is_file():
        raise FileNotFoundError(f"Motion npz does not exist: {motion_npz}")
    if not template_xml.is_file():
        raise FileNotFoundError(f"Template xml does not exist: {template_xml}")
    if not template_ik.is_file():
        raise FileNotFoundError(f"Template ik does not exist: {template_ik}")

    seeds = _build_seed_plan(seed=int(args.seed), num_robots=int(args.num_robots), seed_list=args.robot_seeds)
    print(f"Selected seeds ({len(seeds)}): {seeds}")

    print("[1/5] Loading human motion...")
    human_frames, aligned_fps, human_height = rt.load_human_motion_frames(
        npz_path=motion_npz,
        body_models_dir=body_models_dir,
        target_fps=float(args.target_fps),
    )
    print(
        f"      frames={len(human_frames)} | aligned_fps={_fmt(aligned_fps, 4)} | "
        f"height_est={_fmt(human_height, 3)} m"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    human_frames_cache: Dict[float, List[Dict[str, Tuple[object, object]]]] = {float(aligned_fps): human_frames}  # type: ignore[assignment]
    eval_results: List[Tuple[int, r2h.RobotEval]] = []
    failures: List[Dict[str, str]] = []

    print("[2/5] Generating robots + dynamic IK + retargeting + scoring...")
    for idx, seed in enumerate(seeds):
        robot_dir = output_dir / f"robot_{idx:03d}"
        xml_path = robot_dir / "robot.xml"
        ik_path = robot_dir / "smplx_to_robot.json"
        pkl_path = robot_dir / f"{motion_stem}.pkl"
        metadata_path = robot_dir / "metadata.json"
        robot_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{idx + 1}/{len(seeds)}] {robot_dir.name} | seed={seed}")
        try:
            generate_needed = not bool(args.keep_existing and xml_path.is_file() and ik_path.is_file())
            if generate_needed:
                model = rt.build_random_robot_xml(
                    output_xml_path=xml_path,
                    template_xml=template_xml,
                    seed=int(seed),
                    add_head_joints=not bool(args.disable_head_joints),
                )
                ik_cfg = rt.generate_ik_config(
                    model=model,
                    output_path=ik_path,
                    template_ik_path=template_ik,
                    include_head_task=not bool(args.disable_head_task),
                )
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "seed": int(seed),
                            "robot_dir": str(robot_dir),
                            "nbody": int(model.nbody),
                            "ik1_size": int(len(ik_cfg["ik_match_table1"])),
                            "ik2_size": int(len(ik_cfg["ik_match_table2"])),
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"      generated: nbody={model.nbody}, "
                    f"ik1={len(ik_cfg['ik_match_table1'])}, ik2={len(ik_cfg['ik_match_table2'])}"
                )
            else:
                print("      using existing robot.xml + IK")

            retarget_needed = not bool(args.keep_existing and pkl_path.is_file())
            if retarget_needed:
                robot_key = f"random_humanoid_eval_{int(time.time())}_{idx}"
                rt.retarget_motion(
                    robot_key=robot_key,
                    robot_xml_path=xml_path,
                    ik_config_path=ik_path,
                    human_frames=human_frames,
                    human_height=float(human_height),
                    aligned_fps=float(aligned_fps),
                    save_path=pkl_path,
                    visualize=False,
                    rate_limit=False,
                    record_video=False,
                    video_path=None,
                )
                print(f"      retarget OK -> {pkl_path.name}")
            else:
                print("      using existing pkl")

            result = r2h.evaluate_robot(
                motion_npz=motion_npz,
                body_models_dir=body_models_dir,
                robot_dir=robot_dir,
                motion_stem=motion_stem,
                human_frames_cache=human_frames_cache,  # type: ignore[arg-type]
                enable_dtw=bool(args.enable_dtw),
                max_dtw_frames=int(args.max_dtw_frames),
            )
            eval_results.append((seed, result))
            _save_per_robot_score(robot_dir=robot_dir, seed=seed, result=result)
            print(
                f"      score: overall={_fmt(result.overall_score, 2)} | "
                f"imit={_fmt(result.imitation_score, 2)} | quality={_fmt(result.quality_score, 2)} | "
                f"mpjpe={_fmt(result.mpjpe_cm, 2)}cm"
            )
        except Exception as exc:
            failures.append({"robot_dir": str(robot_dir), "seed": str(seed), "error": str(exc)})
            print(f"      [FAIL] {exc}")

    if not eval_results:
        raise RuntimeError("No valid score was obtained.")

    print("\n[3/5] Final ranking")
    ranked = sorted(eval_results, key=lambda x: x[1].overall_score, reverse=True)
    for rank_idx, (seed, res) in enumerate(ranked, start=1):
        print(
            f"{rank_idx:02d}. {Path(res.robot_dir).name} (seed={seed}) | "
            f"overall={_fmt(res.overall_score, 2)} | "
            f"mpjpe={_fmt(res.mpjpe_cm, 2)}cm | pck10={_fmt(res.pck_10cm, 1)}%"
        )

    print("[4/5] Saving summary...")
    payload = {
        "motion_npz": str(motion_npz),
        "body_models_dir": str(body_models_dir),
        "template_xml": str(template_xml),
        "template_ik": str(template_ik),
        "output_dir": str(output_dir),
        "motion_stem": motion_stem,
        "target_fps_requested": float(args.target_fps),
        "aligned_fps": float(aligned_fps),
        "human_height_est_m": float(human_height),
        "seeds_requested": seeds,
        "num_success": len(ranked),
        "num_failures": len(failures),
        "results_ranked": [
            {
                "seed": int(seed),
                **asdict(res),
            }
            for seed, res in ranked
        ],
        "failures": failures,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"      -> {summary_json}")

    print("[5/5] Comparative visualization")
    if args.visualize_ranked:
        to_visualize = _robot_dirs_with_scores(
            results=ranked,
            top_k=max(0, int(args.visualize_top_k)),
            worst_k=max(0, int(args.visualize_worst_k)),
        )
        if not to_visualize:
            print("      no robots selected for visualization")
        else:
            print(
                "      Original human + robot will be shown (sequential). "
                "Close the window to go to the next."
            )
            for vis_idx, (seed, res) in enumerate(to_visualize, start=1):
                robot_dir = Path(res.robot_dir)
                xml_path = Path(res.xml_path)
                ik_path = robot_dir / "smplx_to_robot.json"
                pkl_path = Path(res.motion_pkl)
                video_path = robot_dir / f"{motion_stem}.scored.mp4"

                print(
                    f"\n      [{vis_idx}/{len(to_visualize)}] {robot_dir.name} (seed={seed}) | "
                    f"overall={_fmt(res.overall_score, 2)} | mpjpe={_fmt(res.mpjpe_cm, 2)}cm"
                )
                print(
                    f"      label: rank={vis_idx}, score={_fmt(res.overall_score, 2)}, "
                    f"imit={_fmt(res.imitation_score, 2)}, quality={_fmt(res.quality_score, 2)}"
                )
                robot_key = f"viz_robot_{vis_idx}_{int(time.time())}"
                rt.retarget_motion(
                    robot_key=robot_key,
                    robot_xml_path=xml_path,
                    ik_config_path=ik_path,
                    human_frames=human_frames,
                    human_height=float(human_height),
                    aligned_fps=float(aligned_fps),
                    save_path=pkl_path,
                    visualize=True,
                    rate_limit=not bool(args.no_rate_limit),
                    record_video=bool(args.record_video),
                    video_path=video_path if bool(args.record_video) else None,
                )
    else:
        print("      skipped (use --visualize-ranked to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()
