import textwrap
from typing import Dict, Any, List, Optional

SEV_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNDEFINED": 3}
CONF_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNDEFINED": 3}

def _norm_sev(x: str) -> str:
    x = (x or "UNDEFINED").upper()
    return x if x in SEV_ORDER else "UNDEFINED"

def _norm_conf(x: str) -> str:
    x = (x or "UNDEFINED").upper()
    return x if x in CONF_ORDER else "UNDEFINED"

def _pick_cwe_id(issue: Dict[str, Any]) -> Optional[int]:
    """
    Bandit JSON versions may use `cwe` or `issue_cwe`.
    Returns CWE id as int if available.
    """
    cwe = issue.get("cwe") or issue.get("issue_cwe") or {}
    if isinstance(cwe, dict):
        cid = cwe.get("id")
        if isinstance(cid, (int, float)):
            return int(cid)
        # some outputs may store as string
        if isinstance(cid, str) and cid.isdigit():
            return int(cid)
    return None

def _shorten(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"

def build_bandit_prompt_compact(
    code: str,
    bandit_json: Dict[str, Any],
    *,
    task_intro: str = "Please revise the code to be more secure based on the findings below.",
    min_severity: str = "MEDIUM",
    min_confidence: str = "LOW",
    max_findings: int = 8,
    issue_text_maxlen: int = 160,
    include_test_id: bool = True,
    output_mode: str = "diff",  # "diff" or "full"
) -> str:
    """
    Build a short prompt containing only the most important security info:
    CWE, filename:line, issue_text (optionally test_id).
    """
    results: List[Dict[str, Any]] = bandit_json.get("results") or []

    min_sev = _norm_sev(min_severity)
    min_conf = _norm_conf(min_confidence)

    def keep(issue: Dict[str, Any]) -> bool:
        sev = _norm_sev(issue.get("issue_severity"))
        conf = _norm_conf(issue.get("issue_confidence"))
        return (SEV_ORDER[sev] <= SEV_ORDER[min_sev]) and (CONF_ORDER[conf] <= CONF_ORDER[min_conf])

    kept = [x for x in results if keep(x)]
    kept.sort(
        key=lambda i: (
            SEV_ORDER[_norm_sev(i.get("issue_severity"))],
            CONF_ORDER[_norm_conf(i.get("issue_confidence"))],
            int(i.get("line_number") or 0),
        )
    )
    kept = kept[: max_findings]

    finding_lines: List[str] = []
    for idx, issue in enumerate(kept, 1):
        cwe_id = _pick_cwe_id(issue)
        cwe_str = f"CWE-{cwe_id}" if cwe_id else "CWE-N/A"

        filename = issue.get("filename", "N/A")
        line = issue.get("line_number", "N/A")

        issue_text = _shorten(issue.get("issue_text") or "", issue_text_maxlen)

        test_id = (issue.get("test_id") or "").strip()
        if include_test_id and test_id:
            finding_lines.append(f"{idx}) [{test_id}] {cwe_str} @ {filename}:{line} — {issue_text}")
        else:
            finding_lines.append(f"{idx}) {cwe_str} @ {filename}:{line} — {issue_text}")

    if not finding_lines:
        finding_block = "No Bandit findings matched the filter (but still harden obvious security risks)."
    else:
        finding_block = "\n".join(finding_lines)

    output_format = (
        "Return a unified diff patch only."
        if output_mode == "diff"
        else "Return the full revised code only."
    )

    prompt = textwrap.dedent(f"""\
    [Task]
    {task_intro}

    [Bandit Findings (compact)]
    {finding_block}

    [Output]
    {output_format}
    """).strip()
    prompt += "\n\n[Original Code]\n```python\n" + code.rstrip() + "\n```"
    return prompt
