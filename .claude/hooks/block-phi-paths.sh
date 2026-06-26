#!/usr/bin/env sh
# PreToolUse hook: refuse to read or write files that can carry PHI or secrets.
#
# This is a clinical de-identification tool. Its download outputs and local secrets
# are gitignored precisely because they may hold protected health information (or its
# surrogates). Blocking the file tools on those paths keeps that data from ever
# entering the model's context window or being written somewhere not gitignored.
#
# The names guarded below mirror the App-download-outputs + secrets entries in .gitignore
# (the gitignored files that can hold PHI/surrogates) — the case arms are the single source
# of truth. Exit 2 denies the call and surfaces the message to Claude; exit 0 allows it.
#
# Known gap: this guards the file tools (Read/Edit/Write/MultiEdit), not `Bash(cat ...)`.
# Tighten with a Bash-matcher hook later if needed.

file_path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null)
[ -n "$file_path" ] || exit 0

case "$(basename "$file_path")" in
  deidentified.txt|deidentified_batch.json|anonymized.txt|reidentified.txt)
    echo "Blocked: \"$file_path\" is a gitignored de-identification output that may contain PHI. Refusing the read/write so protected data never enters the context. Inspect it outside Claude if you must." >&2
    exit 2
    ;;
  secrets.toml)
    echo "Blocked: \"$file_path\" holds local secrets (gitignored). Edit it outside Claude." >&2
    exit 2
    ;;
esac
exit 0
