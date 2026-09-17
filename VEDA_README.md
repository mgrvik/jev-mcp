# Veda Jev MCP (singleton)

Comprehensive TypeSafe/Jev decision-plane MCP staged behind **master-mcp**.

## Based on
- **Primary engine:** `~/veda-converge-diverge` (DecisionService, contracts, episodic store)
- **Vendor reference:** `~/vendor/jev-mcp` (jkudish/jev-mcp) — `jev_verify` / `jev_screen` / `jev_find` parity
- **Review vendor:** `~/vendor/jev-review` (NiazMorshed2007/jev-review) — `jev_review` multi-dimension quality review

## Entry
```
/home/veda/veda-core/.venv/bin/python /home/veda/vendor/jev-mcp/veda_jev_mcp.py
```
Catalog server id: **`veda-jev`**  
Aliases: `jev`, `jev-mcp`, `typesafe`, `decision-plane`, `veda-typesafe`

## TypeSafe key
- File: `~/typesafe-key.md` (line 1)
- Written to `.env`: `~/.env`, `~/veda-converge-diverge/.env`, `~/vendor/jev-mcp/.env`, `~/vendor/jev-review/.env`
- Env: `TYPESAFE_API_KEY` + `JEV_API_KEY` (catalog embeds both)
- Live model: `jev-latest` → server resolves `jev-1.13.0`

## Commands (30)
| Group | Tools |
|-------|--------|
| Vendor jev-mcp | `jev_verify`, `jev_screen`, `jev_find` |
| **jev-review** | `jev_review`, `jev_review_dimensions`, `jev_review_compare`, `jev_review_via_vendor` |
| Health/meta | `jev_health`, `jev_commands` |
| Contracts | `jev_contract_list`, `jev_contract_get`, `jev_context_compile` |
| Decide | `jev_decide`, `jev_advise`, `jev_rank_options` |
| Life | `jev_life_decide`, `jev_life_scenarios` |
| Code/agent | `jev_code_decide`, `jev_tool_select`, `jev_agent_next_action` |
| Bold Action | `jev_bold_action` |
| Episodic graph | `jev_decision_history`, `jev_decision_tree`, `jev_decision_link`, `jev_decision_get`, `jev_decision_recent`, `jev_report_outcome` |
| Live Decision Log | `jev_decision_log_list`, `jev_decision_log_history`, `jev_decision_log_tree` |

## jev_review dimensions
Core: correctness, cognitiveComplexity, readability, modularity, changeability, testQuality, reliability, security  
Conditional: performance, scalability, compatibility, observability  
Plus: coupling, abstractionQuality, projectStructure, duplication, maintainability, consistency, documentation

## Invoke via master-mcp
```text
mcp_exec(server="veda-jev", tool="jev_review",
  args={"task":"...","diff":"...","repositoryContext":"..."})
mcp_exec(server="veda-jev", tool="jev_life_decide",
  args={"scenario":"career","context":{"goal":"...","options":["A","B"]}})
```
