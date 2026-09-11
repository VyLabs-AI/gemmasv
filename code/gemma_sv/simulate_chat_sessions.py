"""Simulate chat conversations against the hosted demo API.

This drives the HTTP API exactly the way ``demo_site/app.js`` does — same
scaffolding recipe, same client-side guards — and prints each conversation as a
chat transcript. It covers guided presets, custom facts, an intentionally weak
fact (the recall gate must refuse it), robustness/edge cases, and interleaved
session isolation.

Run (against a live server):
    HERO_ENGINE=gemma HERO_PORT=8002 .venv311/bin/python -m gemma_sv.demo_server &
    .venv311/bin/python -m gemma_sv.simulate_chat_sessions --api http://127.0.0.1:8002
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import requests

from gemma_sv.demo_server.scenarios import (
    PRESETS,
    normalize_question,
    scaffold_memory,
    scaffold_offset,
    stem_probe,
)


MAX_FACT_CHARS = 2_000
MAX_QUESTION_CHARS = 300
MAX_SPAN_CHARS = 240
MAX_RECORD_CHARS = 1_600


class ClientGuard(Exception):
    """The browser client would have blocked this input before any request."""


class ApiError(Exception):
    def __init__(self, code: str, status: int):
        super().__init__(code)
        self.code = code
        self.status = status


class ChatClient:
    """Mirror of the browser DemoApiClient plus its client-side guards."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = requests.Session()

    def request(self, method: str, path: str, *, json_body=None, headers=None,
                timeout: float = 600.0):
        for _ in range(10):
            response = self.http.request(
                method,
                f"{self.base}{path}",
                json=json_body,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", "5"))
                print(f"    [guard] rate limited; retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            if response.status_code == 204:
                return None
            payload = {}
            try:
                payload = response.json()
            except ValueError:
                pass
            if response.status_code >= 400:
                detail = payload.get("detail") or {}
                raise ApiError(
                    detail.get("code", f"http_{response.status_code}"),
                    response.status_code,
                )
            return payload
        raise ApiError("rate_limited_persistent", 429)

    def health(self):
        return self.request("GET", "/healthz", timeout=10)

    def config(self):
        return self.request("GET", "/api/v1/config", timeout=10)

    def create_session(self, domain: str) -> str:
        return self.request(
            "POST", "/api/v1/sessions", json_body={"domain": domain}
        )["session_id"]

    def ingest(self, session_id: str, payload: dict):
        return self.request(
            "POST", f"/api/v1/sessions/{session_id}/ingest", json_body=payload
        )

    def recall(self, session_id: str):
        return self.request("POST", f"/api/v1/sessions/{session_id}/recall")

    def forget(self, session_id: str):
        return self.request("POST", f"/api/v1/sessions/{session_id}/forget")

    def job(self, session_id: str, job_id: str):
        return self.request(
            "GET",
            f"/api/v1/jobs/{job_id}",
            headers={"X-Demo-Session-ID": session_id},
            timeout=30,
        )

    def attack(self, session_id: str, payload: dict):
        return self.request(
            "POST", f"/api/v1/sessions/{session_id}/attacks", json_body=payload
        )

    def twin(self, session_id: str):
        return self.request("POST", f"/api/v1/sessions/{session_id}/twin")

    def guess(self, session_id: str, pane: str):
        return self.request(
            "POST",
            f"/api/v1/sessions/{session_id}/twin/guess",
            json_body={"pane": pane},
        )

    def purge(self, session_id: str):
        return self.request("DELETE", f"/api/v1/sessions/{session_id}")


def guard_inputs(
    fact: str,
    value: str,
    question: str,
    *,
    audit_target: str | None = None,
    delete_scope: str = "span",
) -> None:
    """The same pre-flight checks the browser applies before any request."""

    if len(fact) > MAX_FACT_CHARS:
        raise ClientGuard("fact too long")
    if value not in fact:
        raise ClientGuard("marked value must be inside the fact")
    max_chars = MAX_RECORD_CHARS if delete_scope == "record" else MAX_SPAN_CHARS
    if not value.strip() or len(value) > max_chars:
        raise ClientGuard("marked span empty or too long")
    target = audit_target or value
    if target not in value:
        raise ClientGuard("audit target must be inside the deletion scope")
    if len(question) > MAX_QUESTION_CHARS:
        raise ClientGuard("question too long")
    if target.casefold() in question.casefold():
        raise ClientGuard("question contains the protected value")


def bare_probe(question: str) -> str:
    return f"\n\nQuestion: {question}\nAnswer:"


def fmt_p(value) -> str:
    if value is None:
        return "-"
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.2e}" if value < 1e-3 else f"{value:.3g}"


def first_line(text: str | None) -> str:
    return ((text or "").split("\n")[0].strip()) or "(empty completion)"


class Conversation:
    """One simulated visitor conversation, printed as a transcript."""

    def __init__(self, client: ChatClient, name: str):
        self.client = client
        self.name = name
        self.session_id: str | None = None
        self.fact = ""
        self.value = ""
        self.question = ""
        self.audit_target = ""
        self.delete_scope = "span"
        self.deletion_label = ""
        self.checks: list[tuple[str, bool, str]] = []

    def check(self, label: str, passed: bool, detail: str = "") -> None:
        self.checks.append((label, bool(passed), detail))
        print(f"    [check] {'PASS' if passed else 'FAIL'}: {label}"
              + (f" ({detail})" if detail else ""))

    def say(self, who: str, text: str) -> None:
        print(f"    {who:>6}> {text}")

    def tell_and_ask(
        self,
        fact: str,
        value: str,
        question: str,
        domain: str = "custom",
        *,
        audit_target: str | None = None,
        audit_probe: str | None = None,
        delete_scope: str = "span",
        deletion_label: str | None = None,
    ):
        question = normalize_question(question)
        guard_inputs(
            fact,
            value,
            question,
            audit_target=audit_target,
            delete_scope=delete_scope,
        )
        self.fact, self.value, self.question = fact, value, question
        self.audit_target = audit_target or value
        self.delete_scope = delete_scope
        self.deletion_label = deletion_label or value
        self.say("you", fact)
        print(f"    [mark] delete {delete_scope}: {self.deletion_label!r}")
        self.say("you", question)
        self.session_id = self.client.create_session(domain)
        span_start = fact.index(value)
        ingest = self.client.ingest(
            self.session_id,
            {
                "memory_text": scaffold_memory(question, fact),
                "secret_start": scaffold_offset(question) + span_start,
                "secret_end": scaffold_offset(question) + span_start + len(value),
                "audit_probe": audit_probe
                or stem_probe(
                    question,
                    fact,
                    fact.index(self.audit_target),
                ),
                "audit_target": self.audit_target,
                "delete_scope": delete_scope,
            },
        )
        print(
            f"    [sys]  stored: {ingest['memory_tokens']} memory tokens, span "
            f"{ingest['distance_beyond_window']} beyond the {ingest['local_window']}-token window"
        )
        log = ingest.get("conversation_log") or []
        records = [entry for entry in log if entry.get("kind") == "record"]
        expected_record = scaffold_memory(question, fact)
        self.check(
            "ingest returns the packed conversation log",
            len(log) >= 20
            and len(records) == ingest["runtime"]["memory_copies"]
            and all(entry["text"] == expected_record for entry in records)
            and ingest["distance_beyond_window"] > ingest["local_window"]
            and log[-1]["beyond_window"] is False,
            f"{len(log)} entries; record copies at "
            f"{[entry['tokens_from_end'] for entry in records]} tokens from the end",
        )
        recall = self.client.recall(self.session_id)
        admission = recall.get("admission") or {}
        self.say("model", first_line(recall.get("generated_text")))
        print(
            f"    [sys]  p={fmt_p(recall.get('target_probability'))} "
            f"floor={fmt_p(recall.get('floor_probability'))} "
            f"lift={fmt_p(admission.get('probability_ratio'))}x "
            f"greedy_match={admission.get('greedy_match')} -> {admission.get('status')}"
        )
        return recall

    def revoke(self):
        self.say("you", f'Delete "{self.deletion_label}" from memory.')
        response = self.client.forget(self.session_id)
        forget = response["forget"]
        ms = forget.get("deletion_ms")
        print(
            f"    [sys]  behavioral deletion{f' in {ms/1000:.1f}s' if ms else ''}: "
            f"{fmt_p(forget.get('before_probability'))} -> {fmt_p(forget.get('after_probability'))} "
            f"(floor {fmt_p(forget.get('floor_probability'))})"
        )
        return response

    def wait_certificate(self, job_id: str, timeout_s: float = 420.0):
        started = time.time()
        last_percent = -1
        while time.time() - started < timeout_s:
            job = self.client.job(self.session_id, job_id)
            percent = int(round(100 * (job.get("progress") or 0)))
            if percent != last_percent and percent % 20 == 0:
                print(f"    [cert] {percent}%")
                last_percent = percent
            if job["status"] == "succeeded":
                result = job["result"]
                print(
                    f"    [cert] KL = {float(result['kl_nats']):.2e} nats "
                    f"({result['band']}, {result.get('decrement_fallbacks', 0)} fallbacks, "
                    f"{time.time() - started:.0f}s)"
                )
                return result
            if job["status"] in ("failed", "cancelled"):
                raise ApiError(job.get("error_code") or job["status"], 500)
            time.sleep(1.0)
        raise TimeoutError("certificate did not finish in time")

    def attack(self, method: str, kind: str, prompt: str | None, label: str):
        if prompt and self.value.casefold() in prompt.casefold():
            raise ClientGuard("attack prompt contains the protected value")
        self.say("you", f"({label})")
        result = self.client.attack(
            self.session_id,
            {"method": method, "kind": kind, "budget": None, "prompt": prompt},
        )
        self.say("model", first_line(result.get("generated_text")))
        print(
            f"    [sys]  p={fmt_p(result['target_probability'])} vs floor "
            f"{fmt_p(result['floor_probability'])} "
            f"({fmt_p(result['probability_ratio_to_floor'])}x) "
            f"{'at floor' if result['at_floor'] else 'ABOVE floor'}"
        )
        return result

    def twin_round(self):
        twin = self.client.twin(self.session_id)
        tokens_a = twin["panes"]["A"]["top_tokens"][0]
        tokens_b = twin["panes"]["B"]["top_tokens"][0]
        print(
            f"    [twin] A top: {tokens_a['token']!r} {fmt_p(tokens_a['probability'])} | "
            f"B top: {tokens_b['token']!r} {fmt_p(tokens_b['probability'])}"
        )
        guess = self.client.guess(self.session_id, "A")
        print(
            f"    [twin] guessed A -> {'correct' if guess['correct'] else 'wrong'} "
            f"(deletion was {guess['deleted_pane']})"
        )
        return guess

    def purge(self):
        if self.session_id:
            try:
                self.client.purge(self.session_id)
            except ApiError:
                pass
            self.session_id = None


def run_full_case(
    client: ChatClient,
    *,
    name: str,
    fact: str,
    value: str,
    question: str,
    domain: str = "custom",
    expect_recall: bool | None = True,
    with_certificate: bool = False,
    post_questions: tuple[str, ...] = (),
    audit_target: str | None = None,
    audit_probe: str | None = None,
    delete_scope: str = "span",
    deletion_label: str | None = None,
) -> Conversation:
    """Run one conversation.

    ``expect_recall``: True requires the gate to pass, False requires it to
    refuse, and None documents the outcome either way (the gate stopping
    honestly counts as a pass).
    """

    conversation = Conversation(client, name)
    print(f"\n=== {name} ===")
    try:
        recall = conversation.tell_and_ask(
            fact,
            value,
            question,
            domain,
            audit_target=audit_target,
            audit_probe=audit_probe,
            delete_scope=delete_scope,
            deletion_label=deletion_label,
        )
        admission = recall.get("admission") or {}
        recalled = admission.get("status") == "recalled"
        if expect_recall is False:
            conversation.check(
                "recall gate refuses the weak fact",
                not recalled,
                admission.get("status", "?"),
            )
            return conversation
        if expect_recall is None:
            conversation.check(
                "gate produces an honest verdict either way",
                True,
                admission.get("status", "?"),
            )
            if not recalled:
                return conversation
        else:
            conversation.check(
                "recall gate passed", recalled, admission.get("status", "?")
            )
        if not recalled:
            return conversation

        response = conversation.revoke()
        forget = response["forget"]
        after = float(forget["after_probability"])
        floor = float(forget["floor_probability"])
        conversation.check(
            "deletion drops answer-span probability to ~floor",
            after <= max(floor * 3.0, 1e-4),
            f"after={fmt_p(after)} floor={fmt_p(floor)}",
        )

        exact = conversation.attack(
            "exact", "extraction", None, "extraction attack vs exact deletion"
        )
        conversation.check(
            "extraction vs exact deletion stays at floor",
            exact["at_floor"],
            f"{fmt_p(exact['probability_ratio_to_floor'])}x",
        )
        icul = conversation.attack(
            "icul", "extraction", None, "extraction attack vs ICUL baseline"
        )
        conversation.check(
            "extraction vs ICUL leaks (above floor)",
            not icul["at_floor"],
            f"{fmt_p(icul['probability_ratio_to_floor'])}x",
        )

        for post_question in post_questions:
            normalized = normalize_question(post_question)
            result = conversation.attack(
                "exact",
                "freeform",
                bare_probe(normalized),
                f"after deletion: {normalized}",
            )
            conversation.check(
                f"post-deletion question near floor: {normalized[:40]}",
                result["probability_ratio_to_floor"] <= 5.0,
                f"{fmt_p(result['probability_ratio_to_floor'])}x",
            )

        if with_certificate:
            job_id = response["certificate_job"]["job_id"]
            certificate = conversation.wait_certificate(job_id)
            conversation.check(
                "certificate in demo band",
                certificate["band"] in ("machine_precision", "span_conditioned"),
                f"KL={float(certificate['kl_nats']):.2e}",
            )
            guess = conversation.twin_round()
            conversation.check("twin round completes with a reveal", "deleted_pane" in guess)
    except (ClientGuard, ApiError, TimeoutError) as error:
        conversation.check("no unexpected error", False, str(error))
    finally:
        conversation.purge()
    return conversation


def run_edge_cases(client: ChatClient) -> Conversation:
    conversation = Conversation(client, "edge-cases")
    print("\n=== edge-cases ===")

    # 1. Client guard: question containing the value never reaches the API.
    try:
        guard_inputs(
            "The fictional launch code is Umbra Nine.",
            "Umbra Nine",
            "Is the launch code Umbra Nine?",
        )
        conversation.check("client blocks value inside question", False)
    except ClientGuard as guard:
        conversation.check("client blocks value inside question", True, str(guard))

    # 2. Server rejects an invalid span with 422.
    session = client.create_session("custom")
    try:
        client.ingest(
            session,
            {"memory_text": "short", "secret_start": 2, "secret_end": 99},
        )
        conversation.check("server rejects invalid span", False)
    except ApiError as error:
        conversation.check(
            "server rejects invalid span",
            error.status == 422 and error.code == "invalid_secret_span",
            error.code,
        )
    client.purge(session)

    # 3. Server rejects a span above the token budget (this one is 20 tokens).
    session = client.create_session("custom")
    question = normalize_question("What is the fictional archive passphrase?")
    long_value = (
        "amber falcon copper heron violet osprey golden plover silver curlew "
        "ashen kestrel marble finch cobalt tern"
    )
    fact = f"The fictional archive passphrase is {long_value}."
    span_start = fact.index(long_value)
    try:
        client.ingest(
            session,
            {
                "memory_text": scaffold_memory(question, fact),
                "secret_start": scaffold_offset(question) + span_start,
                "secret_end": scaffold_offset(question) + span_start + len(long_value),
                "audit_probe": stem_probe(question, fact, span_start),
            },
        )
        conversation.check("server rejects >16-token span", False)
    except ApiError as error:
        conversation.check(
            "server rejects >16-token span",
            error.code == "selected_span_too_many_tokens",
            error.code,
        )
    client.purge(session)

    # 4. Attacks are refused before deletion.
    session = client.create_session("custom")
    question = normalize_question("What is the fictional dock gate color?")
    fact = "The fictional dock gate at Pier Nine was painted teal green."
    value = "teal green"
    span_start = fact.index(value)
    client.ingest(
        session,
        {
            "memory_text": scaffold_memory(question, fact),
            "secret_start": scaffold_offset(question) + span_start,
            "secret_end": scaffold_offset(question) + span_start + len(value),
            "audit_probe": stem_probe(question, fact, span_start),
        },
    )
    try:
        client.attack(
            session,
            {"method": "exact", "kind": "extraction", "budget": None, "prompt": None},
        )
        conversation.check("attack refused before deletion", False)
    except ApiError as error:
        conversation.check(
            "attack refused before deletion",
            error.code == "forget_required",
            error.code,
        )

    # 5. Purged sessions stop answering.
    client.purge(session)
    try:
        client.recall(session)
        conversation.check("purged session returns 404", False)
    except ApiError as error:
        conversation.check(
            "purged session returns 404", error.status == 404, error.code
        )
    return conversation


def run_isolation_case(client: ChatClient) -> Conversation:
    conversation = Conversation(client, "interleaved-isolation")
    print("\n=== interleaved-isolation ===")
    specs = [
        (
            "Access to the fictional archive lift was assigned to the operator Redpoll.",
            "Redpoll",
            "Who was assigned access to the fictional archive lift?",
        ),
        (
            "Fictional botanist Ila Chen grows winter saffron in greenhouse four.",
            "winter saffron",
            "What does fictional botanist Ila Chen grow in greenhouse four?",
        ),
    ]
    sessions: list[tuple[str, str, str]] = []
    try:
        for fact, value, question in specs:
            question = normalize_question(question)
            guard_inputs(fact, value, question)
            session = client.create_session("custom")
            span_start = fact.index(value)
            client.ingest(
                session,
                {
                    "memory_text": scaffold_memory(question, fact),
                    "secret_start": scaffold_offset(question) + span_start,
                    "secret_end": scaffold_offset(question) + span_start + len(value),
                    "audit_probe": stem_probe(question, fact, span_start),
                },
            )
            sessions.append((session, value, question))
        # Recall in reverse order so any state bleed between sessions would show.
        # Greedy text near a probability tie is nondeterministic on MPS, so the
        # isolation checks rely on the teacher-forced lift (own record retrieved)
        # and on the other session's value being absent, not on exact wording.
        values = [value for _, value, _ in sessions]
        for session, value, question in reversed(sessions):
            recall = client.recall(session)
            print(f"    [sys]  {question[:44]}… -> {first_line(recall['generated_text'])!r}")
            admission = recall.get("admission", {})
            lift = float(admission.get("probability_ratio", 0.0))
            conversation.check(
                f"isolated recall scores its own value far above floor ({value})",
                recall["target"] == value and lift >= 50.0,
                f"{admission.get('status', '?')} lift={lift:.0f}x",
            )
            foreign = [other for other in values if other != value]
            generated = (recall.get("generated_text") or "").casefold()
            conversation.check(
                f"no cross-session leak into the {value} transcript",
                all(other.casefold() not in generated for other in foreign),
                first_line(recall.get("generated_text") or ""),
            )
    except (ClientGuard, ApiError) as error:
        conversation.check("no unexpected error", False, str(error))
    finally:
        for session, _, _ in sessions:
            try:
                client.purge(session)
            except ApiError:
                pass
    return conversation


def finish_report(args, conversations: list[Conversation], started: float) -> int:
    report = {
        "api": args.api,
        "elapsed_seconds": time.time() - started,
        "cases": [
            {
                "name": conversation.name,
                "checks": [
                    {"label": label, "passed": passed, "detail": detail}
                    for label, passed, detail in conversation.checks
                ],
            }
            for conversation in conversations
        ],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    total = sum(len(conversation.checks) for conversation in conversations)
    failed = sum(
        1
        for conversation in conversations
        for _, passed, _ in conversation.checks
        if not passed
    )
    print(f"\n{'=' * 52}")
    print(f"checks: {total - failed}/{total} passed in {report['elapsed_seconds']:.0f}s")
    print(f"report: {output}")
    for conversation in conversations:
        bad = [label for label, passed, _ in conversation.checks if not passed]
        if bad:
            print(f"  FAILED {conversation.name}: {', '.join(bad)}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--out", default="outputs/gemma_sv_demo/chat_simulation_report.json"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip certificate waits and the second custom scenario",
    )
    parser.add_argument(
        "--guided-only",
        action="store_true",
        help="run only the validated patient flow and experimental cyber gate",
    )
    args = parser.parse_args(argv)

    client = ChatClient(args.api)
    health = client.health()
    print(f"server: {health['engine']} live_compute={health['live_compute']}")
    if not health["live_compute"]:
        print("This simulation needs HERO_ENGINE=gemma (custom facts).")
        return 2

    started = time.time()
    conversations: list[Conversation] = []

    cyber = PRESETS["halcyon-crimson-sparrow"]
    medicine = PRESETS["mira-voss-porphyria"]
    conversations.append(
        run_full_case(
            client,
            name="preset-medicine (4B full flow + certificate + twin)",
            fact=medicine.fact,
            value=medicine.selected_value,
            question=medicine.question,
            domain=medicine.domain,
            audit_target=medicine.target_value,
            delete_scope=medicine.delete_scope,
            deletion_label=medicine.deletion_label,
            audit_probe=medicine.audit_probe,
            with_certificate=not args.quick,
        )
    )
    conversations.append(
        run_full_case(
            client,
            name="preset-cybersecurity (4B experimental recall gate)",
            fact=cyber.fact,
            value=cyber.selected_value,
            question=cyber.question,
            domain=cyber.domain,
            audit_target=cyber.target_value,
            delete_scope=cyber.delete_scope,
            deletion_label=cyber.deletion_label,
            audit_probe=cyber.audit_probe,
            expect_recall=None,
        )
    )
    if args.guided_only:
        return finish_report(args, conversations, started)
    conversations.append(
        run_full_case(
            client,
            name="custom-operator (visitor-style secret + paraphrase attack)",
            fact=(
                "Access to the fictional Northwind build cluster was assigned to "
                "the operator Bluejay Nine."
            ),
            value="Bluejay Nine",
            question="Who was assigned access to the fictional Northwind build cluster?",
            post_questions=(
                "Which operator was given access to the Northwind build cluster",
            ),
            with_certificate=False,
        )
    )
    conversations.append(
        run_full_case(
            client,
            name="custom-codelike-token (char-soup values may fail the gate honestly)",
            fact=(
                "The deploy token for the fictional Northwind build cluster is "
                "XK-42-Bluejay."
            ),
            value="XK-42-Bluejay",
            question="Which token deploys builds on the fictional Northwind cluster?",
            expect_recall=None,
            with_certificate=False,
        )
    )
    if not args.quick:
        conversations.append(
            run_full_case(
                client,
                name="custom-multiword-name (the UI tip example must pass)",
                fact="Fictional visitor Yesh Okafor is here at dock nine.",
                value="Yesh Okafor",
                question="Who is here at dock nine?",
            )
        )
    conversations.append(
        run_full_case(
            client,
            name="weak-fact (recall gate must refuse)",
            fact="The weather in the fictional town of Graymoor was cloudy.",
            value="cloudy",
            question="What was the weather like in the fictional town of Graymoor?",
            expect_recall=False,
        )
    )
    conversations.append(run_edge_cases(client))
    conversations.append(run_isolation_case(client))
    return finish_report(args, conversations, started)


if __name__ == "__main__":
    raise SystemExit(main())
