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
# of truth, so ADD AN ARM WHENEVER A TAB GAINS A DOWNLOAD. They have drifted once already:
# `policy_anonymized.txt` (the Policy de-ID tab, streamlit_app.py) shipped after this hook
# and stayed readable until the arms were re-synced.
# Exit 2 denies the call and surfaces the message to Claude; exit 0 allows it.
#
# Fails CLOSED: if the path cannot be parsed at all (no python3, malformed/empty stdin) the
# call is denied rather than allowed — a guard that cannot see what it is guarding must not
# wave it through. That is distinct from an empty file_path out of a *successfully parsed*
# payload: non-file tools carry no .tool_input.file_path, so that case still allows.
#
# Known gaps: this guards the file tools (Read/Edit/Write/MultiEdit), not `Bash(cat ...)`,
# and it matches on basename — a symlink, a `.bak` suffix, or a case variant (this repo
# lives on case-insensitive APFS) still reaches the file. Treat it as a guard against
# accidental reads, not as containment.

file_path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null) || {
  echo "Blocked: the PHI-path guard could not parse the tool call (no python3, or malformed input). Failing closed rather than risk letting PHI through unchecked." >&2
  exit 2
}
[ -n "$file_path" ] || exit 0

case "$(basename "$file_path")" in
  deidentified.txt|deidentified_batch.json|anonymized.txt|policy_anonymized.txt|reidentified.txt)
    echo "Blocked: \"$file_path\" is a gitignored de-identification output that may contain PHI. Refusing the read/write so protected data never enters the context. Inspect it outside Claude if you must." >&2
    exit 2
    ;;
  secrets.toml|.env|.env.*)
    echo "Blocked: \"$file_path\" holds local secrets (gitignored). Edit it outside Claude." >&2
    exit 2
    ;;
esac
exit 0
