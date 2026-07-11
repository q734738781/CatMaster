# Nature Downloader In CatMaster

CatMaster uses its runtime-owned `agent-browser` MCP session for browser control. The older standalone CDP proxy and Node download scripts are not part of the active path.

Institutional access is always user-authorized. Configure a real library entry when useful:

```bash
python scripts/configure_school.py infer "https://example.edu/library/resources"
python scripts/configure_school.py url "https://example.edu/library/resources"
python scripts/configure_school.py show
python scripts/configure_school.py health --force
```

Then let the user complete login in the controlled Chrome session. The agent must stop for passwords, CAPTCHA, QR, OTP, security warnings, or unclear consent.

Downloaded files stay under the active workspace, are ingested with `ingest_literature_files`, and are queried with `query_literature_corpus`. Final selected DOIs are normalized in one `finalize_citations` batch.
