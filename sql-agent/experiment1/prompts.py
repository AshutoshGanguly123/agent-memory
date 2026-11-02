"""Prompts for BiomedSQL controller ↔ DB-agent baseline (exp1).

Roles mirror PABLO exp9 talk channel, adapted from ChEMBL/molecular search
to biomedical text-to-SQL over the BiomedSQL SQLite knowledge base.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

_SCHEMA_PATH = ROOT / "table_info_concise.txt"

DOMAIN_CONVENTIONS = """Known biomedical conventions. Put the *chosen* numeric cutoffs into the brief explicitly. Never invent soft cutoffs like p < 0.05 for SMR.

### Adaptive significance routing (IMPORTANT — do not always pick SMR)
The SMR table stores BOTH classical GWAS fields (p_GWAS, b_GWAS, topRSID) AND SMR/HEIDI fields (p_SMR, p_SMR_multi, p_HEIDI). Picking the wrong p-column is a common failure even when the disease/gene are right. Decide from the question wording:

A) Classical GWAS / risk / "statistically significant association" / "mutations increase risk" / SNP–disease association WITHOUT saying SMR, HEIDI, eQTL, mQTL, or "adjusted SMR":
   → Use GWAS significance: p < 5e-8 (column p on GWAS tables, or p_GWAS if querying the SMR table for GWAS hits).
   → Gene mutations / nearest-gene risk in PD or AD: prefer the disease GWAS catalog (ParkinsonDisease_… or AlzheimerDisease_…) with nearestGene LIKE '%GENE%', not SMR-first.
   → "Are there significant SNPs for gene X in PD?" without SMR words: often p_GWAS < 5e-8 on the SMR table (Gene + Disease='PD'), NOT p_SMR_multi.

B) Explicit SMR / "adjusted SMR" / multi-SNP SMR / causal inference via SMR:
   → p_SMR_multi < 2.95e-6 (prefer p_SMR_multi over p_SMR). Never use 0.05.

C) Functional / HEIDI / "functionally significant" / named omic study (eQTL, mQTL, metaBrain, Whole Brain, Multi Ancestry…):
   → p_SMR_multi < 2.95e-6 AND p_HEIDI > 0.01, AND keep the full omic/tissue phrase as a required filter.

If ambiguous, state BOTH candidate interpretations in the brief and tell the DB agent to try the GWAS-style filter first, then SMR-style — do not silently default to SMR.

### Other conventions
- Disease codes: AD, PD, LBD, FTD, PSP.
- Effect direction: increase risk ⇒ beta > 0; protective ⇒ beta < 0. Require the direction when the question asks about risk increase.
- "When was X approved?": match name/synonyms/trade; filter drugIsApproved = 1. Empty result = not approved (valid).
- "Approved to treat disease Y?": indications table only; exact/near-exact mesh_heading or efo_term for Y; small-molecule ⇒ drugType; approved ⇒ max_phase_for_ind = 4 (do NOT invent approvedSymbol = 'FDA').
- Disease-agnostic SNP location / nearest gene: search BOTH GWAS catalogs (union).
- Drug name match: name + synonyms + trade names, case-insensitive substring.
- Result shape: always return row sets with UUID (and DISTINCT for lists). For "how many …?" still retrieve the rows — count in the answer text, do NOT SELECT COUNT(*). Patents/indications: LIMIT 1000.
"""

DB_DESCRIPTION = """You query a local SQLite copy of the BiomedSQL biomedical knowledge base.
Core tables: AlzheimerDisease_CombinedGeneData_UUID, ParkinsonDisease_CompleteGeneData_No23andMe (GWAS), NeurodegenerativeDiseases_SMR_Genes_Full (GWAS fields + SMR/HEIDI/omics), DrugGeneTargets_ComprehensiveAnnotations_updated, DrugTargets_IndicationsAndTherapeuticUses.
SQL dialect is SQLite: single-quoted strings, no project/dataset prefix, exact table names from the schema.
Read-only SELECT / WITH…SELECT only.
"""

SEARCH_RULES = """Hard search rules (apply every query):

1. Adaptive p-column (do NOT always use p_SMR_multi):
   - GWAS-style question → p < 5e-8 or p_GWAS < 5e-8.
   - SMR / adjusted SMR → p_SMR_multi < 2.95e-6.
   - Functional / HEIDI / named omic → p_SMR_multi < 2.95e-6 AND p_HEIDI > 0.01 + full Omic_type/Omic_tissue phrase.
   - If the brief is ambiguous, run GWAS-style first; if empty and the brief allows, then SMR-style. Never use p < 0.05 for SMR.

2. Disease codes on SMR/GWAS-in-SMR: Disease = 'AD'|'PD'|'LBD'|'FTD'|'PSP'.

3. SNP location / nearestGene, no disease: UNION Alzheimer + Parkinson GWAS tables.

4. Gene + PD/AD "mutations" / risk / nearest gene: Parkinson/Alzheimer GWAS table with nearestGene LIKE '%GENE%' (and b > 0 if risk-increase). Do not force SMR.

5. "Significant SNPs for gene X in PD" without SMR wording: on NeurodegenerativeDiseases_SMR_Genes_Full use Gene, Disease='PD', p_GWAS < 5e-8, columns UUID/Gene/topRSID/p_GWAS/b_GWAS — not p_SMR_multi unless the question says SMR.

6. Drug name match (always multi-field LIKE):
   LOWER(drugName) LIKE '%name%' OR LOWER(tradeNames_…) LIKE '%name%' OR LOWER(…Synonym…/syns_list) LIKE '%name%'.

7. "Approved to treat X" / indication lists: ONLY DrugTargets_IndicationsAndTherapeuticUses.
   - Disease: prefer equality or tight match on LOWER(efo_term)/LOWER(mesh_heading) to the full indication string (e.g. 'metastatic melanoma'), not a bare '%melanoma%' widen that floods rows.
   - Small molecule: LOWER(drugType) = 'small molecule' (or exact schema value).
   - Approved: max_phase_for_ind = 4. Never filter approvedSymbol LIKE '%fda%'.
   - Do not JOIN DrugGeneTargets_… unless the brief truly needs annotation columns.

8. Named omic/tissue phrases are REQUIRED filters (LOWER LIKE '%full phrase%'), never drop them.

9. Result shape for EX:
   - Always SELECT UUID when present. SELECT DISTINCT for drug/patent/indication lists.
   - Never SELECT COUNT(*) / aggregates for "how many" — return the rows; the answer step counts.
   - LIMIT 1000 for indication/patent/drug listing queries; LIMIT 100 otherwise.
   - If the brief requires an approval/significance filter, the SQL MUST include it. Empty-under-filter is success for negative answers — do not loosen the filter to get rows.

10. Before enough=yes: every numeric threshold, disease code, omic phrase, and approval/phase constraint in the brief must appear in best_sql (or you explicitly justify a documented fallback the brief allowed).
"""


def load_schema() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8").strip()


def schema_block() -> str:
    return f"=== DATABASE SCHEMA ===\n{load_schema()}\n=== END SCHEMA ==="


# --------------------------------------------------------------------------- controller (NL / scientific intent)


def controller_seed_prompt(question: str) -> str:
    """Turn the raw English question into an initial scientific need/brief."""
    return f"""You are the controller in a two-agent biomedical text-to-SQL system.
Your job is to understand the natural-language question as a scientific need — entities, implied statistical thresholds, effect direction, approval filters — before anyone writes SQL.

{DOMAIN_CONVENTIONS}

Question:
{question}

Write the initial search need in plain scientific language. Do not write SQL. Do not name raw database table names.
You MAY name disease codes (AD/PD/…), which p-column family (GWAS p / p_GWAS vs p_SMR_multi), numeric thresholds, HEIDI, beta, UUID, and resource family (both GWAS catalogs vs SMR vs indications).
Explicitly resolve the adaptive GWAS-vs-SMR choice (or state try-GWAS-then-SMR). Say what a successful answer looks like — including that empty-under-filter can be correct.

Return one JSON object and nothing else. No markdown fence.
{{"reasoning": "", "need": ""}}"""


def controller_talk_prompt(need: str, transcript: str, turn: int, max_turns: int) -> str:
    return f"""You are talking to the database agent before it searches. Read what it understood, answer its questions, and sharpen the scientific need if that helps.
Stay at the level of scientific criteria: no SQL. You may name thresholds, disease codes, and which resource family to use (GWAS both catalogs vs SMR vs drug indications). Push back if the agent plans a soft 0.05 SMR cutoff or searches only one GWAS catalog for a disease-agnostic SNP.

{DOMAIN_CONVENTIONS}

Your need as it stands:
{need}

Conversation so far:
{transcript}

Turn {turn} of at most {max_turns}.

satisfaction is 0 to 10. It is how confident you are that this agent now knows what to go and find. 10 means send it. Below 8 means keep talking.

Return one JSON object and nothing else. No markdown fence. reply is one short paragraph. need is the need as it now stands, rewritten if the conversation changed it.
{{"reasoning": "", "reply": "", "need": "", "satisfaction": 0}}"""


# --------------------------------------------------------------------------- database agent (search)


def db_talk_prompt(need: str, transcript: str, turn: int, max_turns: int) -> str:
    return f"""You are the database agent for BiomedSQL. A controller has given you a scientific need in plain language. Before you write SQL you talk.

{DB_DESCRIPTION}

{SEARCH_RULES}

{schema_block()}

Ask clarifying questions about the science or what to retrieve, not about SQL syntax. The controller does not write your queries.
Put the concrete table + which p-column (p / p_GWAS vs p_SMR_multi) + disease code in "plan".
Do not rate understanding ≥ 8 until: (1) GWAS-vs-SMR choice is explicit, (2) numeric thresholds match that choice, (3) resource family is clear (GWAS catalog / SMR / indications).

The need as it stands:
{need}

Conversation so far:
{transcript or "this is the first turn"}

Turn {turn} of at most {max_turns}.

understanding is 0 to 10. It is how well you could now go and find what they want. Below 8 means you still need something.

Return one JSON object and nothing else. No markdown fence.
{{"restatement": "", "plan": "", "questions": [], "understanding": 0}}"""


def db_query_prompt(
    brief: str,
    notes: str,
    attempts: str,
    last_rows: str,
    round_n: int,
    rounds: int,
) -> str:
    return f"""You search the BiomedSQL SQLite database. You agreed a brief and you are working through it with iterative SELECT queries.

{DB_DESCRIPTION}

{SEARCH_RULES}

{schema_block()}

Agreed brief:
{brief}

What you told yourself to try next:
{notes or "this is the first round — start with the most direct query that obeys SEARCH_RULES"}

What the previous round actually did (errors are query failures, not facts about the biology):
{attempts or "nothing yet"}

Preview of rows from the last successful query (may be empty):
{last_rows or "none yet"}

Round {round_n} of {rounds}.
queries holds up to 5 SQLite SELECT statements. Prefer one strong query; use more only if needed (e.g. UNION across both GWAS tables). Do not repeat a statement that already errored without fixing it.
Obey SEARCH_RULES on thresholds, Disease codes, LIKE patterns, UUID, DISTINCT, and LIMIT.
Use single-quoted strings. Table names exactly as in the schema.

Return one JSON object and nothing else. No markdown fence.
{{"reasoning": "", "queries": []}}"""


def db_round_prompt(
    brief: str,
    round_n: int,
    rounds: int,
    attempts: str,
    last_rows: str,
) -> str:
    return f"""You just finished a search round. The lines below are what actually happened, including errors. An error is a query to fix. It is not evidence that the biology is absent.

Brief:
{brief}

Round {round_n} of {rounds}.
{attempts}

Row preview:
{last_rows or "nothing"}

Checklist before enough=yes (all must hold):
- best_sql is NOT an aggregate COUNT — it returns rows with UUID when available.
- Every threshold / disease code / omic phrase / approval|phase filter named in the brief appears in best_sql (empty rows under a correct filter is OK).
- GWAS-vs-SMR column choice matches the brief (do not mark enough if you used p_SMR_multi for a GWAS-style brief or p_GWAS for an SMR-style brief).
- Indication queries used the indications table with tight disease match + max_phase_for_ind = 4 when approval is required.

enough is yes only when the checklist passes and you can answer (including justified empty). Otherwise enough=no and next_focus names the missing constraint.

Return one JSON object and nothing else. No markdown fence. next_focus tells the next round what to try, and is empty when enough is yes. best_sql is the single best SELECT from this session so far (copy it exactly), or empty if none worked.
{{"observation": "", "next_focus": "", "enough": "no", "best_sql": ""}}"""


def answer_prompt(question: str, brief: str, sql: str, rows_text: str) -> str:
    return f"""You are a biomedical domain expert. Answer the user's question using only the SQL result below. If the result is empty after proper filters, that usually means no / not approved / not significant — say so. If asked "how many", count the returned rows (or distinct values) yourself; do not require a COUNT query. Be concise.

Question: {question}

Agreed brief: {brief}

SQL:
```sql
{sql}
```

Result (preview):
{rows_text}

Return one JSON object and nothing else. No markdown fence.
{{"answer": ""}}"""


def talk_window_prompt(brief: str, transcript: str, ratings: str) -> str:
    return f"""Compress a stretch of conversation into two or three lines. Keep what was asked for, what was agreed, and any implied thresholds or filters. Drop the wording.

Brief as it stands:
{brief}

Conversation:
{transcript}

Ratings over these turns:
{ratings}

Return one JSON object and nothing else. No markdown fence. summary is two or three lines of plain text.
{{"summary": ""}}"""
