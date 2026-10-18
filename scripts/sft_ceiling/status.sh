#!/bin/bash
# Print uwrun status for every sftceil run in the ledger (latest per name). Usage: status.sh [PATTERN]
PAT=${1:-sftceil}
cut -f2 ~/.claude/uw/ledger.tsv | grep -- "$PAT" | sort -u | while read -r n; do
  timeout 90 ~/.claude/bin/uwrun status "$n" 2>&1 | tail -1
done
