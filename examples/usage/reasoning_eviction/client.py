"""Token-ID client for multi-turn benchmarks; the server owns pruning/replay."""

import argparse
import json
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class ReasoningSessionClient:
    def __init__(self, base_url, tokenizer, *, api_key=None, timeout=600):
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.base_url + path,
            data=json.dumps(body or {}).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error

    def open(self, window=None):
        return self._post("/reasoning_sessions/open", {"window": window})["session_id"]

    def control(self, session_id, operation):
        if operation not in ("evict", "cancel", "close"):
            raise ValueError("Unknown session operation")
        return self._post(f"/reasoning_sessions/{session_id}/{operation}")

    def build_turn(self, messages, *, reasoning_mode="think", reasoning_priming=""):
        """Render only new user/tool/system messages and one open assistant.

        Return both exact token IDs and their message structure. Never scan
        quoted user/tool tags to infer roles, and never strip history here.
        """
        if reasoning_mode not in ("think", "think_fast", "think_faster"):
            raise ValueError("Unknown Horizon reasoning mode")

        def encode(text):
            return self.tokenizer.encode(text, add_special_tokens=False)

        if any(
            role not in ("user", "tool", "system") or not isinstance(content, str)
            for role, content in messages
        ):
            raise ValueError("Supply only new user/tool/system text messages")
        tokens, boundaries = [], []
        for role, content in [*messages, ("assistant", None)]:
            start = len(tokens)
            tokens.extend(encode("<|ifm|im_start|>"))
            tokens.extend(encode(role))
            body_start = len(tokens)
            tokens.extend(encode("\n"))
            completed = content is not None
            if completed:
                tokens.extend(encode(content))
                tokens.extend(encode("<|ifm|im_end|>"))
            else:
                tokens.extend(encode(f"<ifm|{reasoning_mode}>"))
                tokens.extend(encode("\n" + reasoning_priming))
            boundaries.append(
                dict(
                    role=role,
                    start=start,
                    body_start=body_start,
                    end=len(tokens),
                    completed=completed,
                )
            )
        return {"input_ids": tokens, "messages": boundaries}

    def generate(
        self,
        session_id,
        *,
        input_ids,
        messages,
        request_id=None,
        sampling_params=None,
        return_logprob=False,
    ):
        """Reuse request_id and identical arguments to retry an uncertain response."""
        return self._post(
            f"/reasoning_sessions/{session_id}/generate",
            dict(
                request_id=request_id or uuid.uuid4().hex,
                input_ids=input_ids,
                messages=messages,
                sampling_params=sampling_params
                or {"temperature": 0, "max_new_tokens": 1024},
                return_logprob=return_logprob,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--window", default="8")
    parser.add_argument("--force-replay", action="store_true")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.revision, trust_remote_code=False
    )
    client = ReasoningSessionClient(args.url, tokenizer)
    sid = client.open(None if args.window == "full" else int(args.window))
    try:
        for index, content in enumerate(
            ("What is 17 plus 25?", "Add 3 to that result.")
        ):
            if index and args.force_replay:
                client.control(sid, "evict")
            result = client.generate(
                sid,
                **client.build_turn([("user", content)]),
                request_id=f"turn-{index}",
            )
            print(json.dumps(result, indent=2))
            if not result["meta_info"]["reasoning_eviction"]["completed"]:
                raise RuntimeError(
                    "Assistant was truncated; continue with empty input_ids/messages before adding a turn"
                )
    finally:
        client.control(sid, "close")


if __name__ == "__main__":
    main()
