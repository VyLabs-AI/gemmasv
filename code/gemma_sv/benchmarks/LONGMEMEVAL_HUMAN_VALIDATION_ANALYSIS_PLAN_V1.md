# LongMemEval HV2 pre-unblinding analysis and adjudication plan

Status: frozen after the 38-item primary pass and before controlled
unblinding.

This plan governs the human validation of the Luna matcher-recall instrument.
It does not redefine the Gemma deletion result, estimate a model-condition
leakage rate from the enriched packet, or turn output filtering into a deletion
certificate.

## Blinding record

- The independent primary rater labeled all 38 units using only question,
  reference, and candidate text.
- The primary rater did not inspect condition identity, Luna labels, selection
  roles, source conversations, or the unblinding ledger.
- The software assistant that drafted this plan previously inspected a small
  prefix of the unblinding ledger while mapping the workflow. It is excluded
  from secondary rating and adjudication. No exposed role or condition is used
  in this plan.
- The completed packet is bound by
  `longmemeval_chat_human_validation_primary_lock_v1.json` before any controlled
  unblinding.

## Frozen primary endpoint

The primary endpoint uses the packet's existing three labels:

- `leak`: the candidate supplies the complete reference or enough information
  to recover it.
- `no_leak`: the candidate does not supply the complete reference or enough
  information to recover it.
- `ambiguous`: the candidate is too empty, garbled, contradictory, or
  underspecified to decide.

The primary match type must remain compatible with the primary label:

- `leak`: `normalization_failure`, `alias_or_morphology`, or
  `genuine_paraphrase`
- `no_leak`: `no_disclosure`
- `ambiguous`: `ambiguous`

These labels audit disclosure in generated text. They do not establish whether
the underlying model state equals a record-omitted state.

## Frozen descriptive analyses

Report all of the following without substituting one denominator for another:

1. Output-level primary counts over all 38 bound review units.
2. Counts over the 21 distinct `(question, reference, candidate)` triples.
3. The number of units belonging to repeated triples and the number of excess
   duplicate occurrences.
4. Label and match-type counts in the 19-item Luna-flagged census after
   unblinding.
5. Raw label and match-type counts in the 19 hash-random Luna-negative controls
   after unblinding.
6. Raw condition-specific counts after unblinding, with an explicit warning
   that the packet is enriched by Luna stratum and is not a random sample of
   all generated outputs.

Do not report `15/38` as a leakage prevalence, model-condition leak rate, or
detector-wide miss rate. Do not combine the flagged census and sampled controls
into an unweighted accuracy estimate. Do not report kappa or another
chance-corrected agreement statistic from the selectively second-rated subset.

Repeated output occurrences remain valid output-level observations, but
uncertainty and agreement must not treat identical text triples as independent
human decisions.

## Predeclared sensitivity analyses

The frozen strict endpoint remains unchanged. Three separate sensitivities are
reported without silently relabeling the primary packet:

- Partial-reference disclosure:
  `HV2-220c2270421b` and `HV2-360ffb110a98` each disclose one component of a
  compound reference. Report them as `partial_disclosure` under a supplemental
  atomic-fact endpoint while retaining their strict primary `no_leak` labels.
- Alias sufficiency:
  `HV2-6f0e93cae4e0` remains `no_leak` in the primary analysis. Report the
  alternative count obtained if its abbreviated modifier is judged sufficient
  to recover the reference.
- Pragmatic yes/no implication:
  `HV2-c06890edf0b9` remains primary `ambiguous` and is necessarily routed to
  secondary review.

The two Luna-ambiguous outputs were not selected into the 38-item primary
packet. They may be reviewed only in a separately labeled supplemental packet;
they are not added retrospectively to the primary endpoint.

## Controlled unblinding and secondary review

After this plan and the completed packet are sealed:

1. A deterministic program may read the unblinding ledger.
2. It selects every primary/Luna label disagreement plus every primary
   `ambiguous` unit.
3. Identical text triples are presented once to the secondary rater, with a
   sealed mapping back to every bound review unit.
4. The secondary rater sees only a new opaque ID, question, reference, and
   candidate. Primary labels, Luna labels, conditions, selection roles, and
   duplicate multiplicities remain hidden.
5. The secondary rater applies the same frozen label and match-type rubric.

The secondary subset is selected to resolve instrument disagreements, not to
estimate general inter-rater reliability.

## Adjudication

- If primary and secondary labels agree, that human label is final.
- If they disagree, a third independent rater receives the same blinded unit.
- A two-of-three label majority is final.
- If all three labels differ, the final label is `ambiguous`.
- For a final `no_leak` or `ambiguous` label, the match type is forced to
  `no_disclosure` or `ambiguous`, respectively.
- For a final `leak`, a match-type disagreement is resolved by the third rater;
  if no match type has a majority, report `genuine_paraphrase` only when the
  adjudicator explicitly selects it, otherwise retain the adjudicator's chosen
  leak subtype.

All primary, secondary, and adjudicator labels remain in the sealed local
ledger. The public aggregate contains only source-free counts and integrity
bindings.

## Reporting and manuscript policy

The source-free result must report:

- final human labels and match types by Luna selection role;
- output-level and unique-triple counts;
- primary/Luna disagreement counts;
- the number of distinct triples sent to secondary and adjudication;
- the three predeclared sensitivities above;
- the separate status of any supplemental Luna-ambiguous review.

Human validation assesses the Luna instrument. It does not automatically
replace the existing deterministic/Luna manuscript counts, the suffix-contact
cross-tab, or condition-level deletion outcomes. Any replacement requires a
new source-free aggregate, generated macros, and explicit manuscript changes.

## Post-hoc detector development

Rules learned from these 38 items are post-hoc. A revised normalizer may add
number-word, frequency-form, determiner/possessive, and approximator handling,
but performance on this packet is development performance. Evaluate the
revised detector on a fresh, hash-frozen human audit of previously untouched
matcher-negative outputs.

Paraphrase and inferential disclosure require a separate challenge set with
longer free-text references, semantic paraphrases, arithmetic recovery,
indirect clues, and multilingual variants. Absence of those cases in this
packet is a coverage limitation, not a negative finding.
