"""Persistent, Taipei-dated daily digest state; SMTP is never retried blindly.

Only public archive-bound content is staged outside site/. The workflow commits
each transition before performing the next irreversible operation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, time as clock_time
from pathlib import Path
from zoneinfo import ZoneInfo

from crossref_ingestion import save_ingestion_history
from recommendation_history import save_history
from site_builder import build_daily_payload, publish_daily_payload

TZ = ZoneInfo("Asia/Taipei")
ROOT = Path(".daily-digest")
SITE = Path("site")
PHASES = {"prepared", "empty", "sending", "accepted", "sent"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def state_path(root, day):
    return Path(root) / day.isoformat() / "state.json"


def archive_path(site, day):
    return Path(site) / "data" / "daily" / (day.isoformat() + ".json")


def phase(root, site, day):
    path = state_path(root, day)
    archived = archive_path(site, day).is_file()
    if not path.exists():
        # Migration: the old production workflow generated archives after SMTP.
        return "legacy" if archived else "new"
    state = read_json(path)
    if state.get("date") != day.isoformat() or state.get("phase") not in PHASES:
        raise ValueError("Invalid digest state: " + str(path))
    value = state["phase"]
    if archived and value in {"prepared", "sending", "empty"}:
        raise RuntimeError("Digest state conflicts with an existing archive; investigate before sending")
    if value == "sent" and not archived:
        raise RuntimeError("Sent digest is missing its archive; restore from repository history")
    return value


def write_state(folder, day, value, **extra):
    write_json(Path(folder) / "state.json", {
        "date": day.isoformat(), "phase": value,
        "updated_at": datetime.now(TZ).isoformat(), **extra,
    })


def prepare_empty(folder, day):
    if (Path(folder) / "state.json").exists():
        raise RuntimeError("Refusing to replace existing digest state")
    write_state(folder, day, "empty")


def prepare_digest(folder, day, papers, observed, html):
    folder = Path(folder)
    if (folder / "state.json").exists():
        raise RuntimeError("Refusing to replace existing digest state")
    folder.mkdir(parents=True, exist_ok=True)
    # These identities also end up in the public ingestion ledger. No
    # recipient list, credentials or private Zotero corpus is serialized.
    identities = [{key: getattr(p, key, None) for key in ("doi", "title")}
                  for p in observed]
    write_json(folder / "archive.json", build_daily_payload(papers, day))
    write_json(folder / "observed.json", identities)
    (folder / "email.html").write_text(html, encoding="utf-8")
    write_state(folder, day, "prepared")


def seconds_until_publish(now):
    """Before 07:15, prepare only. From 07:15 wait until 08:00; after, send."""
    if now.tzinfo is None:
        raise ValueError("Expected timezone-aware datetime")
    local = now.astimezone(TZ)
    if local.time() < clock_time(7, 15):
        return None
    target = datetime.combine(local.date(), clock_time(8), tzinfo=TZ)
    return max(0.0, (target - local).total_seconds())


def claim(root, site, day, run_id, attempt):
    if phase(root, site, day) != "prepared":
        raise RuntimeError("Only a prepared digest can be claimed")
    write_state(state_path(root, day).parent, day, "sending",
                run_id=run_id, attempt=attempt)


def deliver(root, site, day, run_id, attempt, send):
    if phase(root, site, day) != "sending":
        raise RuntimeError("Missing durable sending claim")
    folder = state_path(root, day).parent
    state = read_json(folder / "state.json")
    if (state.get("run_id"), state.get("attempt")) != (run_id, attempt):
        raise RuntimeError("SMTP outcome uncertain; do not resend automatically")
    # If SMTP raises, keep the durable sending claim. A network disconnect can
    # occur AFTER acceptance. Only an operator may reconcile this ambiguity.
    send((folder / "email.html").read_text(encoding="utf-8"), day)
    write_state(folder, day, "accepted", run_id=run_id, attempt=attempt)


def finalize(root, site, day):
    if phase(root, site, day) != "accepted":
        raise RuntimeError("Only an SMTP-accepted digest can be archived")
    folder = state_path(root, day).parent
    payload = read_json(folder / "archive.json")
    if payload.get("date") != day.isoformat():
        raise ValueError("Prepared archive has the wrong digest date")
    observed = read_json(folder / "observed.json")
    # These operations are safe to repeat using the same staged payload.
    if observed:
        save_ingestion_history(observed, str(site))
    save_history(payload["papers"], str(site), today=day)
    publish_daily_payload(payload, str(site))
    write_state(folder, day, "sent")
    for name in ("archive.json", "observed.json", "email.html"):
        (folder / name).unlink()


def recover_accepted(root, site):
    """Recover previous or current days after an interrupted archive commit."""
    for path in sorted(Path(root).glob("*/state.json")):
        day = date.fromisoformat(path.parent.name)
        value = phase(root, site, day)
        if value == "accepted":
            finalize(root, site, day)
        elif value == "sending":
            print("Unresolved SMTP outcome for", day, "; operator reconciliation required.")


def output(**values):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
        for key, value in values.items():
            out.write(f"{key}={value}\n")


def smtp_settings():
    from construct_email import parse_receivers
    settings = [os.environ.get(k, "") for k in (
        "SENDER", "RECEIVER", "SENDER_PASSWORD", "SMTP_SERVER", "SMTP_PORT")]
    if not all(settings):
        raise ValueError("SMTP configuration is incomplete")
    parse_receivers(settings[1])
    port = int(settings[4])
    if not 0 < port < 65536:
        raise ValueError("Invalid SMTP port")
    return (*settings[:4], port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("recover", "plan", "ready", "validate",
                                             "claim", "send", "finalize"))
    parser.add_argument("day", nargs="?", type=date.fromisoformat)
    args = parser.parse_args()
    now = datetime.now(TZ)
    day = args.day or now.date()
    if args.command == "recover":
        recover_accepted(ROOT, SITE)
        return
    if args.command == "plan":
        value = phase(ROOT, SITE, day)
        print("Taipei digest day:", day, "phase:", value)
        output(day=day.isoformat(), phase=value)
        if value == "sending":
            raise RuntimeError("SMTP outcome uncertain: verify delivery before resetting the claim")
        return
    if args.command == "ready":
        value = phase(ROOT, SITE, day)
        delay = seconds_until_publish(now)
        ready = value == "prepared" and day == now.date() and delay is not None
        if ready and delay:
            print(f"Waiting {delay:.0f}s until 08:00 Asia/Taipei", flush=True)
            time.sleep(delay)
        # Do not send a previous day's prepared digest after midnight.
        ready = ready and datetime.now(TZ).date() == day
        output(send="true" if ready else "false")
        return
    if args.command == "validate":
        smtp_settings()
        return
    run_id = os.environ["GITHUB_RUN_ID"]
    attempt = os.environ["GITHUB_RUN_ATTEMPT"]
    if args.command == "claim":
        if day != now.date() or now.time() < clock_time(8):
            raise RuntimeError("Claim outside today's publication window")
        claim(ROOT, SITE, day, run_id, attempt)
    elif args.command == "send":
        from construct_email import send_email
        settings = smtp_settings()
        deliver(ROOT, SITE, day, run_id, attempt,
                lambda html, day: send_email(*settings, html, digest_date=day))
    elif args.command == "finalize":
        finalize(ROOT, SITE, day)


if __name__ == "__main__":
    main()
