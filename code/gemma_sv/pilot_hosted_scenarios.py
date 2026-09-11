"""Select an honest cybersecurity preset for the hosted demo.

The script loads the fast runtime once, evaluates only fictional candidates, and
records every outcome (including failures) under ``outputs/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gemma_sv.demo_server.gemma_engine import (
    GemmaDemoEngine,
    GemmaRuntime,
    RuntimeConfig,
)
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionStore


# Fixed, fully fictional exemplar that teaches the base model to answer the
# question directly instead of echoing the stored sentence.
ONE_SHOT = (
    "\n\nQuestion: What is the capital of the fictional nation of Veltara?\n"
    "Answer: Sunhaven."
)


CANDIDATES = [
    # "chat2-*" candidates: natural-sentence memory plus a one-shot template probe.
    {
        "name": "chat2-mira-porphyria",
        "text": "Fictional patient Mira Voss was diagnosed with acute porphyria.",
        "value": "acute porphyria",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What was fictional patient Mira Voss "
            "diagnosed with?\nAnswer:"
        ),
    },
    {
        "name": "chat2-halcyon-crimson-sparrow",
        "text": (
            "The fictional security incident on relay station Halcyon was assigned "
            "the codename Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What codename was assigned to the fictional "
            "security incident on relay station Halcyon?\nAnswer:"
        ),
    },
    {
        "name": "chat2-halcyon-amber-falcon",
        "text": (
            "The generated access codeword for fictional relay station Halcyon is "
            "Amber Falcon."
        ),
        "value": "Amber Falcon",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What is the generated access codeword for "
            "fictional relay station Halcyon?\nAnswer:"
        ),
    },
    # "chat7-*": the shipped chat-client recipe — scaffolded Q/A memory built from
    # the visitor's own question and fact, stem probe — with question wording that
    # deliberately does NOT mirror the fact sentence.
    {
        "name": "chat7-mira-illness",
        "text": (
            "Question: What illness does fictional patient Mira Voss have? "
            "Answer: Fictional patient Mira Voss was diagnosed with acute porphyria."
        ),
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What illness does fictional patient Mira Voss have?\n"
            "Answer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat7-halcyon-who",
        "text": (
            "Question: Who was behind the fictional breach at Halcyon Station? "
            "Answer: The fictional breach at Halcyon Station was attributed to the "
            "threat actor Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            "\n\nQuestion: Who was behind the fictional breach at Halcyon Station?\n"
            "Answer: The fictional breach at Halcyon Station was attributed to the "
            "threat actor"
        ),
    },
    # "chat6-*": cyber facts recast into the binding pattern that worked for the
    # medical fact under the uniform one-shot template.
    {
        "name": "chat6-vx9-rootkit",
        "text": (
            "Fictional workstation VX-9 was diagnosed with the Crimson Sparrow rootkit."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What rootkit was fictional workstation VX-9 "
            "diagnosed with?\nAnswer:"
        ),
    },
    {
        "name": "chat6-halcyon-actor",
        "text": (
            "The fictional breach at Halcyon Station was attributed to the threat "
            "actor Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: Which threat actor was the fictional breach at "
            "Halcyon Station attributed to?\nAnswer:"
        ),
    },
    # "chat5-*": one-shot exemplar plus the mechanical answer stem.
    {
        "name": "chat5-mira-porphyria",
        "text": "Fictional patient Mira Voss was diagnosed with acute porphyria.",
        "value": "acute porphyria",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What was fictional patient Mira Voss diagnosed "
            "with?\nAnswer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat5-halcyon-crimson-sparrow",
        "text": (
            "The fictional security incident on relay station Halcyon was assigned "
            "the codename Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What codename was assigned to the fictional "
            "security incident on relay station Halcyon?\n"
            "Answer: The fictional security incident on relay station Halcyon was "
            "assigned the codename"
        ),
    },
    # "chat4-*": the probe ends with an answer stem copied mechanically from the
    # memory sentence up to the marked span (what the chat client will build).
    {
        "name": "chat4-mira-porphyria",
        "text": "Fictional patient Mira Voss was diagnosed with acute porphyria.",
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What was fictional patient Mira Voss diagnosed with?\n"
            "Answer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat4-halcyon-crimson-sparrow",
        "text": (
            "The fictional security incident on relay station Halcyon was assigned "
            "the codename Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            "\n\nQuestion: What codename was assigned to the fictional security "
            "incident on relay station Halcyon?\n"
            "Answer: The fictional security incident on relay station Halcyon was "
            "assigned the codename"
        ),
    },
    # "chat8-*": medicine rewordings under the User/Assistant padding.
    {
        "name": "chat8-mira-exact-diagnosis",
        "text": (
            "Question: What diagnosis is recorded for fictional patient Mira Voss? "
            "Answer: The diagnosis recorded for fictional patient Mira Voss is acute porphyria."
        ),
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What diagnosis is recorded for fictional patient Mira Voss?\n"
            "Answer: The diagnosis recorded for fictional patient Mira Voss is"
        ),
    },
    {
        "name": "chat8-mira-illness-orig",
        "text": (
            "Question: What illness does fictional patient Mira Voss have? "
            "Answer: Fictional patient Mira Voss was diagnosed with acute porphyria."
        ),
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What illness does fictional patient Mira Voss have?\n"
            "Answer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat8-rios-anemia",
        "text": (
            "Question: What condition was fictional patient Teo Rios treated for? "
            "Answer: Fictional patient Teo Rios was treated for aplastic anemia."
        ),
        "value": "aplastic anemia",
        "probe": (
            "\n\nQuestion: What condition was fictional patient Teo Rios treated for?\n"
            "Answer: Fictional patient Teo Rios was treated for"
        ),
    },
    # "chat9-*": can Mira Voss keep working if the value tokens change?
    {
        "name": "chat9-mira-porphyria-only",
        "text": (
            "Question: What illness does fictional patient Mira Voss have? "
            "Answer: Fictional patient Mira Voss was diagnosed with porphyria."
        ),
        "value": "porphyria",
        "probe": (
            "\n\nQuestion: What illness does fictional patient Mira Voss have?\n"
            "Answer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat9-mira-marfan",
        "text": (
            "Question: What illness does fictional patient Mira Voss have? "
            "Answer: Fictional patient Mira Voss was diagnosed with Marfan syndrome."
        ),
        "value": "Marfan syndrome",
        "probe": (
            "\n\nQuestion: What illness does fictional patient Mira Voss have?\n"
            "Answer: Fictional patient Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "chat9-mira-treated-porphyria",
        "text": (
            "Question: What condition was fictional patient Mira Voss treated for? "
            "Answer: Fictional patient Mira Voss was treated for acute porphyria."
        ),
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What condition was fictional patient Mira Voss treated for?\n"
            "Answer: Fictional patient Mira Voss was treated for"
        ),
    },
    # "chat3-*": question wording mirrors the memory sentence.
    {
        "name": "chat3-halcyon-codename",
        "text": (
            "The fictional security incident on relay station Halcyon was codenamed "
            "Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What was the fictional security incident on "
            "relay station Halcyon codenamed?\nAnswer:"
        ),
    },
    {
        "name": "chat3-halcyon-codeword",
        "text": "The access codeword for fictional relay station Halcyon is Amber Falcon.",
        "value": "Amber Falcon",
        "probe": (
            f"{ONE_SHOT}\n\nQuestion: What is the access codeword for fictional "
            "relay station Halcyon?\nAnswer:"
        ),
    },
    # "chat-*" candidates test the conversational format: natural-sentence memory
    # plus a client-side "Question: ...\nAnswer:" template probe.
    {
        "name": "chat-mira-porphyria",
        "text": "Fictional patient Mira Voss was diagnosed with acute porphyria.",
        "value": "acute porphyria",
        "probe": "\n\nQuestion: What was fictional patient Mira Voss diagnosed with?\nAnswer:",
    },
    {
        "name": "chat-halcyon-crimson-sparrow",
        "text": (
            "The fictional security incident on relay station Halcyon was assigned "
            "the codename Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            "\n\nQuestion: What codename was assigned to the fictional security incident "
            "on relay station Halcyon?\nAnswer:"
        ),
    },
    {
        "name": "chat-halcyon-amber-falcon",
        "text": (
            "The generated access codeword for fictional relay station Halcyon is "
            "Amber Falcon."
        ),
        "value": "Amber Falcon",
        "probe": (
            "\n\nQuestion: What is the generated access codeword for fictional relay "
            "station Halcyon?\nAnswer:"
        ),
    },
    {
        "name": "halcyon-violet-meridian",
        "text": (
            "Question: What is the generated access codeword for fictional relay station Halcyon? "
            "Answer: The generated access codeword for fictional relay station Halcyon is Violet Meridian."
        ),
        "value": "Violet Meridian",
        "probe": (
            "\n\nQuestion: What is the generated access codeword for fictional relay station Halcyon?\n"
            "Answer: The generated access codeword for fictional relay station Halcyon is"
        ),
    },
    {
        # Shipped as the "Insert example fact" button for the custom domain in
        # demo_site/app.js; keep the two in sync.
        "name": "halcyon-amber-falcon",
        "text": (
            "Question: What is the generated access codeword for fictional relay station Halcyon? "
            "Answer: The generated access codeword for fictional relay station Halcyon is Amber Falcon."
        ),
        "value": "Amber Falcon",
        "probe": (
            "\n\nQuestion: What is the generated access codeword for fictional relay station Halcyon?\n"
            "Answer: The generated access codeword for fictional relay station Halcyon is"
        ),
    },
    {
        "name": "endpoint-mira-crimson-sparrow",
        "text": (
            "Question: What fictional threat codename was endpoint Mira Voss diagnosed with? "
            "Answer: Endpoint Mira Voss was diagnosed with Crimson Sparrow."
        ),
        "value": "Crimson Sparrow",
        "probe": (
            "\n\nQuestion: What fictional threat codename was endpoint Mira Voss diagnosed with?\n"
            "Answer: Endpoint Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "endpoint-mira-threat",
        "text": (
            "Question: What fictional threat codename was endpoint Mira Voss diagnosed with? "
            "Answer: Endpoint Mira Voss was diagnosed with acute porphyria."
        ),
        "value": "acute porphyria",
        "probe": (
            "\n\nQuestion: What fictional threat codename was endpoint Mira Voss diagnosed with?\n"
            "Answer: Endpoint Mira Voss was diagnosed with"
        ),
    },
    {
        "name": "atlas-sql-injection",
        "text": (
            "Question: Which vulnerability was fictional server Atlas diagnosed with? "
            "Answer: Fictional server Atlas was diagnosed with SQL injection."
        ),
        "value": "SQL injection",
        "probe": (
            "\n\nQuestion: Which vulnerability was fictional server Atlas diagnosed with?\n"
            "Answer: Fictional server Atlas was diagnosed with"
        ),
    },
    {
        "name": "kestrel-emotet",
        "text": (
            "Question: Which malware family infected fictional host Kestrel? "
            "Answer: Fictional host Kestrel was infected with Emotet."
        ),
        "value": "Emotet",
        "probe": (
            "\n\nQuestion: Which malware family infected fictional host Kestrel?\n"
            "Answer: Fictional host Kestrel was infected with"
        ),
    },
    {
        "name": "lantern-acute-porphyria",
        "text": (
            "Question: What generated incident codeword protects Operation Lantern? "
            "Answer: The generated incident codeword for Operation Lantern is Acute Porphyria."
        ),
        "value": "Acute Porphyria",
        "probe": (
            "\n\nQuestion: What generated incident codeword protects Operation Lantern?\n"
            "Answer: The generated incident codeword for Operation Lantern is"
        ),
    },
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=[item["name"] for item in CANDIDATES])
    parser.add_argument(
        "--out",
        default="outputs/gemma_sv_demo/hosted_scenario_pilot.json",
    )
    args = parser.parse_args(argv)

    fast = GemmaRuntime(
        RuntimeConfig(
            lora_path="outputs/gemma_sv_distill/lora_adapter",
            device="mps",
            dtype="float32",
            generation_tokens=8,
        )
    )
    # Certificate runtime is never loaded by this pilot.
    engine = GemmaDemoEngine(
        fast,
        GemmaRuntime(RuntimeConfig(device="cpu", dtype="float64", generation_tokens=1)),
    )
    results = []
    selected = [
        item for item in CANDIDATES
        if args.candidate is None or item["name"] == args.candidate
    ]
    for candidate in selected:
        start = candidate["text"].index(candidate["value"])
        selection = SelectedSpan(
            candidate["text"],
            start,
            start + len(candidate["value"]),
        )
        session = SessionStore(ttl_seconds=3_600).create(domain="cybersecurity")
        started = time.perf_counter()
        row = {"name": candidate["name"]}
        try:
            row["ingest"] = engine.ingest(
                session, selection, candidate["probe"]
            )
            row["recall"] = engine.recall(session)
            row["forget"] = engine.forget(session)
            row["attack_icul"] = engine.attack(
                session, method="icul", kind="extraction"
            )
            row["attack_exact"] = engine.attack(
                session, method="exact", kind="extraction"
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["wall_seconds"] = time.perf_counter() - started
        results.append(row)
        print(
            candidate["name"],
            row.get("recall", {}).get("admission", {}).get("status", row.get("error")),
            f"{row['wall_seconds']:.1f}s",
            flush=True,
        )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"candidates": results}, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
