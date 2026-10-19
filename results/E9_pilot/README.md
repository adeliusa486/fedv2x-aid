# E9 pilot (validation splits only, seed 0, 18 Oct 2026)

Question: does anchoring each cloud round on the server's labelled data (`--server-anchor`, no extra WAN) let FedV2X-Aid beat the round-0 head, and is any gain due to the aids rather than to the extra server training?

Final validation mean R@P80 after 50 edge rounds (round-0 head: 0.2007):
- FedV2X-Aid + server anchor: 0.2106
- server-only control (same server steps, no aids): 0.2093
- E1 reference curves: FedV2X-Aid 0.2032, SemiFL-style 0.2141

Decision: the anchored run's gain over round 0 (+0.010) is almost entirely explained by the server-only control (+0.009); the aids add about 0.001, within the round-to-round noise of the curves (up to 0.03). E9 was therefore not pre-registered or run on test, and nothing from this pilot is claimed in the paper. The test split was never read.
