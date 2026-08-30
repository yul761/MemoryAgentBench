"""StateCore as a memory method for MemoryAgentBench.

StateCore (github.com/yul761/StateCore) is an auditable memory engine: writes go through a
deterministic pipeline, a corrected fact supersedes its predecessor on a recorded chain
(`supersededBy`), and retirement/discards are logged rather than silent. It is driven here
through its published MCP front end -- the wrapper spawns

    npx -y statecore-mcp@0.5.0 --data <fresh temp dir>

and speaks JSON-RPC over stdio (newline-delimited, per the MCP stdio transport). Nothing is
installed into this venv and no service needs starting by hand; the only requirement is Node
>= 20 on PATH. The version is pinned so a run is reproducible from this file alone
(STATECORE_MCP_SPEC overrides, for testing a newer release without editing it).

ZERO MODEL CALLS ON THE MEMORY SIDE. Every other memory agent in this harness spends LLM calls
on extraction or consolidation. StateCore's note path is deterministic: a write that reads as a
revision of an active fact supersedes it in place (short-token-preserving similarity, so
"deadline is May 3" vs "deadline is May 4" replaces rather than accumulates), and retrieval is
lexical (ASCII words + CJK bigrams) over facts and events. The only LLM in the loop is the
shared reader that every method uses. Whatever score this row gets is therefore the floor of
the engine's LLM-assisted mode, bought at zero memory-side token cost -- that asymmetry is the
point of the row, and it is stated here so nobody reads the comparison as like-for-like on
spend.

NORMALIZED INPUT. Identical to the knowl/agentmemory rows: the parsed fact list, in context
order, one record per write, via `parse_fact_lines` (reused from methods.agentmemory). Same
records, no system-specific extraction. Facts over the note path's 500-char cap (rare in these
datasets) are stored as events instead -- still retrievable, just outside the supersession
machinery.

ISOLATION. Every run gets a fresh --data directory (its own SQLite file), so runs cannot see
each other; the directory is a tempdir and is left for the OS to clean.

READER. Mirrors `_handle_bm25_rag` via the shared knowl helpers, exactly like the agentmemory
row: same query extraction, "Memory i:" labels, same system template. Retrieval contents are
the active facts recall returns (already relevance-ranked and budget-packed by the engine),
then raw events, truncated to retrieve_num.
"""

import json
import os
import subprocess
import tempfile
import time

from methods.agentmemory import parse_fact_lines

DEFAULT_SPEC = "statecore-mcp@0.5.0"
PROTOCOL_VERSION = "2025-06-18"


class StateCoreMcpClient:
    """Minimal MCP-over-stdio client. Requests are sequential, so plain blocking reads on the
    child's stdout are enough; the first `npx -y` run downloads the package and generates its
    database client, which can take a minute -- later runs start in ~2s."""

    def __init__(self, data_dir, spec):
        self.proc = subprocess.Popen(
            ["npx", "-y", spec, "--data", data_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit: the server prints "[statecore-mcp] ready over stdio" there
            text=True,
            bufsize=1,
        )
        self._next_id = 0
        self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "memoryagentbench-statecore", "version": "1"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _request(self, method, params):
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("statecore-mcp exited before replying (is Node >= 20 on PATH?)")
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if message.get("id") != request_id:
                continue  # notifications / unrelated traffic
            if "error" in message:
                raise RuntimeError(f"statecore-mcp error: {message['error']}")
            return message["result"]

    def call_tool(self, name, arguments):
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text = result["content"][0]["text"]
        if result.get("isError"):
            raise RuntimeError(f"statecore-mcp tool {name} failed: {text}")
        return json.loads(text)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


class StateCoreClient:
    def __init__(self, spec):
        self.chunks = []
        self.flushed = False
        self.facts = 0
        self.superseded = 0
        self.data_dir = tempfile.mkdtemp(prefix="mab_statecore_")
        self.mcp = StateCoreMcpClient(self.data_dir, spec)

    def add(self, text):
        self.chunks.append(text)

    def flush(self):
        """Write every parsed fact, in context order. Idempotent, like the agentmemory row.

        Order is the only recency signal the task provides; StateCore's revision matcher is what
        turns "later similar fact" into supersession rather than accumulation.
        """
        if self.flushed:
            return {"facts": self.facts, "superseded": self.superseded}

        facts = parse_fact_lines("".join(self.chunks))
        superseded = 0
        for fact in facts:
            if len(fact) <= 500:
                result = self.mcp.call_tool("remember", {"text": fact})
                if result.get("superseded") is not None:
                    superseded += 1
            else:
                # Over the note cap: stored as an event -- retrievable, but outside supersession.
                self.mcp.call_tool("remember", {"text": fact[:2000], "consolidate": True})

        self.facts = len(facts)
        self.superseded = superseded
        self.flushed = True
        print(f"\nstatecore flush: {self.facts} facts, {superseded} superseded at write\n")
        return {"facts": self.facts, "superseded": superseded}

    def query(self, text, k):
        result = self.mcp.call_tool("recall", {"query": text, "maxChars": 16000})
        contents = []
        for fact in result.get("factRegistry") or []:
            content = fact.get("content")
            if content:
                contents.append(content)
        for event in result.get("events") or []:
            content = event.get("content")
            if content:
                contents.append(content)
        return contents[:k]


def initialize_statecore_agent(agent, agent_config=None):
    config = agent_config or {}
    agent.retrieve_num = config["retrieve_num"]
    agent.context = ""
    agent.agent_start_time = time.time()

    spec = os.environ.get("STATECORE_MCP_SPEC", DEFAULT_SPEC)
    agent.statecore = StateCoreClient(spec)
    print(f"\n\nstatecore ({spec}) embedded at {agent.statecore.data_dir}\n\n")


def handle_statecore_agent(agent, message, memorizing, query_id, context_id):
    """Mirror `_handle_bm25_rag`: same query extraction, same reader assembly."""
    from methods.knowl import build_reader_messages, format_retrieval_memory_string
    from utils.templates import get_template

    if memorizing:
        agent.statecore.add(message)
        return "Memorized"

    start_time = time.time()
    stats = agent.statecore.flush()
    memory_construction_time = time.time() - start_time

    retrieval_query = agent._extract_retrieval_query(message)
    contents = agent.statecore.query(retrieval_query, agent.retrieve_num)
    retrieval_memory_string = format_retrieval_memory_string(contents)

    system_message = get_template(agent.sub_dataset, "system", agent.agent_name)
    format_message = build_reader_messages(retrieval_memory_string, message, system_message)

    response = agent._create_oai_client().chat.completions.create(
        model=agent.model,
        messages=format_message,
        temperature=agent.temperature,
        max_tokens=agent.max_tokens if "gpt-4" in agent.model else None,
    )

    query_time_len = time.time() - start_time - memory_construction_time
    print(f"\nstatecore stats: {stats}\n")

    return agent._create_standard_response(
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        memory_construction_time,
        query_time_len,
    )
