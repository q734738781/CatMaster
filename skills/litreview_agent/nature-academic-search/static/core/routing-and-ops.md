# Source routing and operations

Use `web_search` for initial discovery and the controlled browser when source
inspection, JavaScript rendering, or a user-authorized institutional route is
needed. For a selected paper, treat access as unknown until the DOI or publisher
page has been opened in the controlled Chrome browser. An institutional network,
proxy, or authorized profile/session may expose full-text HTML or PDF directly;
do not infer lack of entitlement from metadata or the absence of an open-access
link. Existing workspace attachments and lawful open-access copies remain valid
alternative routes.

Ingest selected full text and retrieve compact page-level evidence. Keep large
candidate tables, full pages, and document text in workspace artifacts rather
than repeating them in the parent context. Normalize only the final DOI set in
one `finalize_citations` call.

Report specific search, access, parsing, and unresolved-identifier failures.
Only report missing entitlement after a direct browser attempt shows a login
wall or permission denial.
Never bypass access controls or ask for credentials, cookies, or OTP codes.
