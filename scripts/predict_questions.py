#!/usr/bin/env python3
"""
Fetch unpredicted questions from the platform and run predictions using MiroFlow agent.

Usage:
    python scripts/predict_questions.py --config config/agent_quickstart.yaml
    python scripts/predict_questions.py --config config/agent_quickstart.yaml --once
    python scripts/predict_questions.py --config config/agent_quickstart.yaml --interval 1800
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import dotenv
import requests
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Platform client
# ---------------------------------------------------------------------------

PLATFORM_BASE_URL = os.getenv("PLATFORM_BASE_URL")
PLATFORM_TOKEN = os.getenv("PLATFORM_TOKEN")
SUBMIT_TOKEN = os.getenv("SUBMIT_TOKEN")

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Referer": f"{PLATFORM_BASE_URL}/" if PLATFORM_BASE_URL else "",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    ),
    "authorization": PLATFORM_TOKEN,
}

SUBMIT_HEADERS = {
    "Authorization": f"Bearer {SUBMIT_TOKEN}" if SUBMIT_TOKEN else "",
    "Content-Type": "application/json",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def fetch_unpredicted_questions(
    page: int = 1,
    page_size: int = 10000,
    days_back: int = 7,
    exact_previous_day: bool = False,
) -> list[dict]:
    """Fetch questions that have not been predicted yet."""
    now = datetime.now(timezone.utc)
    if exact_previous_day:
        previous_day = (now - timedelta(days=1)).date()
        start = datetime.combine(previous_day, datetime.min.time(), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        created_from = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        created_to = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    else:
        created_to = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        created_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    url = (
        f"{PLATFORM_BASE_URL}/api/questions/search"
        f"?keyword=&created_from={created_from}&created_to={created_to}"
        f"&page={page}&page_size={page_size}"
    )

    try:
        resp = requests.get(url, headers=HEADERS, verify=False, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error("Failed to fetch questions: %s", e)
        return []

    data = resp.json()

    # Support both list response and paginated {"items": [...]} response
    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        questions = data.get("items") or data.get("data") or data.get("questions") or []
    else:
        questions = []

    # Filter to unpredicted questions.
    # The platform uses a status field; treat anything that is not "predicted"
    # (or equivalent) as unpredicted.
    PREDICTED_STATUSES = {"predicted", "completed", "done", "finished"}
    unpredicted = [
        q for q in questions
        if str(q.get("status", "")).lower() not in PREDICTED_STATUSES
    ]

    log.info(
        "Fetched %d questions, %d unpredicted (page %d)",
        len(questions), len(unpredicted), page,
    )
    return unpredicted


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

def _build_agent_and_cfg(config_path: str, output_dir: str):
    """Load config and build agent. Returns (agent, cfg, model_name)."""
    # Import here so the module can be imported without hydra initialised
    import hydra
    import omegaconf
    from config import load_config
    from miroflow.agents import build_agent_from_config
    from miroflow.logging.task_tracer import get_tracer

    cfg = load_config(config_path, f"output_dir={output_dir}")
    agent = build_agent_from_config(cfg)

    # Best-effort model name extraction
    try:
        model_name = cfg.main_agent.llm.model_name
    except Exception:
        model_name = "unknown"

    tracer = get_tracer()
    tracer.set_log_path(output_dir)

    return agent, cfg, model_name


async def _run_prediction(agent, cfg, question: dict) -> dict:
    """Run the agent on a single question and return the structured result."""
    from miroflow.benchmark.eval_utils import Task
    from miroflow.benchmark.task_runner import run_single_task

    q_id = str(question.get("id", uuid.uuid4()))
    content = question.get("content") or question.get("question") or question.get("title") or ""

    # 拼接选项空间
    answer_space = question.get("answer_space", "")
    if answer_space:
        try:
            options = json.loads(answer_space) if isinstance(answer_space, str) else answer_space
            options_str = "、".join(f"{k}: {v}" for k, v in options.items())
            q_text = f"{content}\n\n候选答案：{options_str}"
        except Exception:
            q_text = f"{content}\n\n候选答案：{answer_space}"
    else:
        q_text = content

    # 强制要求先搜索最新新闻再作答
    q_text = (
        f"{q_text}\n\n"
        "【重要指令】\n"
        "1. 你必须先使用搜索工具查询与本问题相关的最新新闻和信息，禁止直接输出结论。\n"
        "2. 至少执行一次新闻搜索后，再根据搜索结果给出答案。\n"
        "3. 在最终回答中，明确说明你查到了哪些关键证据，以及基于这些证据得出答案的推理过程。"
    )

    task = Task(
        task_id=q_id,
        task_question=q_text,
        ground_truth="",
    )

    execution_cfg = cfg.get("benchmark", {}).get("execution", {})

    result = await run_single_task(
        cfg=cfg,
        agent=agent,
        task=task,
        pass_at_k=1,
        max_retry=execution_cfg.get("max_retry", 1),
        evaluator=None,
        exceed_max_turn_summary=execution_cfg.get("exceed_max_turn_summary", False),
        prompt_manager=getattr(agent, "prompt_manager", None),
    )

    # Extract answer and supporting info from the agent result
    raw_answer = result.model_boxed_answer or ""

    # model_response 是 AgentContext（dict子类），直接用 .get()
    response_dict = result.model_response if isinstance(result.model_response, dict) else {}
    if not response_dict and result.attempts:
        last_attempt = result.attempts[-1]
        raw = getattr(last_attempt, "model_response", {})
        response_dict = raw if isinstance(raw, dict) else {}

    # reason = 最后一轮 assistant 的 Response preview（message_history 最后一条 role=assistant）
    message_history = response_dict.get("message_history", [])
    reason = ""
    for msg in reversed(message_history):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        msg_content = msg.get("content", "")
        if isinstance(msg_content, str):
            reason = msg_content.strip()
        elif isinstance(msg_content, list):
            parts = []
            for block in msg_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            reason = "\n".join(parts).strip()
        if reason:
            break

    # evidence = 搜索/工具调用获取的实时内容，从 message_history 中提取 tool 结果
    evidence = _extract_tool_evidence(response_dict)

    # answer 只保留选项部分（第一个标点前），去掉 agent 附带的解释
    answer = raw_answer
    for sep in ["；", "，", "——", ";", "\n"]:
        if sep in raw_answer:
            answer = raw_answer.split(sep, 1)[0].strip()
            break

    return {
        "id": q_id,
        "question": content,
        "model_name": cfg.main_agent.llm.get("model_name", "unknown"),
        "answer": answer,
        "reason": reason,
        "confidence": _estimate_confidence(result),
        "evidence": evidence,
    }


def _extract_tool_evidence(response_dict: dict) -> list:
    """递归扫描 message_history，提取所有搜索和爬取工具结果，多轮合并去重。"""
    message_history = response_dict.get("message_history", [])
    evidence_list = []
    seen_urls = set()

    def _scan(obj):
        if isinstance(obj, str):
            if "organic" not in obj and "extracted_info" not in obj:
                return
            try:
                data = json.loads(obj)
            except Exception:
                return
            if not isinstance(data, dict):
                return

            # google_search 结果：organic 列表，content 用 snippet
            if "organic" in data:
                for item in data["organic"]:
                    url = item.get("link", "")
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    evidence_list.append({
                        "url": url,
                        "content": item.get("snippet") or item.get("title") or "",
                    })

            # scrape_and_extract_info 结果：用完整的 extracted_info
            if data.get("success") and data.get("extracted_info") and data.get("url"):
                url = data["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    evidence_list.append({
                        "url": url,
                        "content": data["extracted_info"],
                    })
                else:
                    # 已存在则用更完整的 extracted_info 替换 snippet
                    for e in evidence_list:
                        if e["url"] == url:
                            e["content"] = data["extracted_info"]
                            break

        elif isinstance(obj, dict):
            for v in obj.values():
                _scan(v)
        elif isinstance(obj, list):
            for item in obj:
                _scan(item)

    _scan(message_history)
    return evidence_list


def _estimate_confidence(result) -> int:
    """置信度，返回 0-100 的整数。"""
    if not result.attempts:
        return 20
    last = result.attempts[-1]
    if getattr(last, "is_valid_box", False):
        return 85
    if result.model_boxed_answer:
        return 60
    return 20


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def submit_prediction(result: dict) -> bool:
    """提交预测结果到平台，返回是否成功。"""
    url = f"{PLATFORM_BASE_URL}/api/agent-predictions/submit"
    try:
        resp = requests.post(url, json=result, headers=SUBMIT_HEADERS, verify=False, timeout=30)
        if resp.status_code in (200, 201):
            log.info("Submitted prediction id=%s, status=%d", result.get("id"), resp.status_code)
            return True
        else:
            log.error("Submit failed id=%s, status=%d, body=%s",
                      result.get("id"), resp.status_code, resp.text[:200])
            return False
    except requests.RequestException as e:
        log.error("Submit request error id=%s: %s", result.get("id"), e)
        return False

def save_results(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"predictions_{ts}.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info("Saved %d results to %s", len(results), out_file)
    return out_file


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_once(
    config_path: str,
    output_dir: str,
    days_back: int,
    limit: int = 0,
    exact_previous_day: bool = False,
) -> list[dict]:
    questions = fetch_unpredicted_questions(days_back=days_back, exact_previous_day=exact_previous_day)
    if not questions:
        log.info("No unpredicted questions found.")
        return []

    if limit > 0:
        questions = questions[:limit]
        log.info("Limiting to %d question(s) for testing.", limit)

    log.info("Building agent from config: %s", config_path)
    agent, cfg, model_name = _build_agent_and_cfg(config_path, output_dir)
    log.info("Using model: %s", model_name)

    results = []
    for i, q in enumerate(questions, 1):
        q_id = q.get("id", "?")
        q_text = (q.get("content") or q.get("question") or q.get("title") or "")[:80]
        log.info("[%d/%d] Predicting question id=%s: %s...", i, len(questions), q_id, q_text)
        try:
            result = asyncio.run(_run_prediction(agent, cfg, q))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            submit_prediction(result)
        except Exception as e:
            log.error("Failed to predict question %s: %s", q_id, e)
            results.append({
                "id": str(q_id),
                "question": q_text,
                "model_name": model_name,
                "answer": "",
                "reason": f"Prediction failed: {e}",
                "confidence": 0,
                "evidence": [],
            })

    save_results(results, Path(output_dir) / "predictions")
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict unpredicted platform questions")
    parser.add_argument(
        "--config",
        default="config/agent_quickstart.yaml",
        help="Path to agent config yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/predictions",
        help="Directory to write prediction logs",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1800,
        help="Poll interval in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit instead of looping",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=7,
        help="How many days back to search for questions (default: 7)",
    )
    parser.add_argument(
        "--exact-previous-day",
        action="store_true",
        help="Fetch only questions created in the previous UTC day window",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only predict the first N questions (0 = no limit, useful for testing)",
    )
    args = parser.parse_args()

    dotenv.load_dotenv()

    if not PLATFORM_BASE_URL or not PLATFORM_TOKEN or not SUBMIT_TOKEN:
        missing = [
            name
            for name, value in {
                "PLATFORM_BASE_URL": PLATFORM_BASE_URL,
                "PLATFORM_TOKEN": PLATFORM_TOKEN,
                "SUBMIT_TOKEN": SUBMIT_TOKEN,
            }.items()
            if not value
        ]
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    if args.once:
        run_once(args.config, args.output_dir, args.days_back, args.limit, args.exact_previous_day)
        return

    log.info("Starting prediction loop (interval=%ds). Press Ctrl+C to stop.", args.interval)
    while True:
        try:
            run_once(args.config, args.output_dir, args.days_back, args.limit, args.exact_previous_day)
        except KeyboardInterrupt:
            log.info("Interrupted, exiting.")
            sys.exit(0)
        except Exception as e:
            log.error("Unexpected error in run_once: %s", e)

        log.info("Sleeping %d seconds until next run...", args.interval)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Interrupted, exiting.")
            sys.exit(0)


if __name__ == "__main__":
    main()
