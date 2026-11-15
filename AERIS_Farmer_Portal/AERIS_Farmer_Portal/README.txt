AERIS FARMER PORTAL — LOCAL DEMO
================================

WHAT THIS IS
A separate, local web page for showing a scientist-approved AERIS crop rotation in simple language. It does NOT modify the Scientist Workbench, prediction engine, or approved JSON file.

START IT
1. Extract this folder anywhere, for example E:\TrialShit\AERIS_Farmer_Portal
2. Open PowerShell in that folder.
3. Run:
   py launch_farmer_portal.py
   (or: python launch_farmer_portal.py)
4. Your browser opens http://127.0.0.1:8765
5. Stop the server with Ctrl+C in PowerShell.

HOW TO USE IT
1. In Scientist Workbench, approve a rotation as usual.
2. Open aeris\data\processed\approved_recommendations.json
3. Copy the complete file contents.
4. Paste into the portal and click "Show my recommendation".
5. Switch to Marathi using the top-right button.

NOTES
- This is a local demonstration. It is not public hosting and is accessible only on this computer.
- The page processes pasted JSON in the browser; it does not upload it to a server.
- It accepts the current AERIS approval format: a JSON object keyed by cell ID, with fields such as target, rotation, crop_names, explanation, evidence_status and fitness_score.
- The sample button demonstrates the layout; it is not a real farm recommendation.
- Farmer-facing reasons are deliberately cautious. AERIS scores are heuristic screening scores, not yield or income predictions.
- The current approved JSON does not contain farm-specific planting dates, irrigation quantities, seed rates, fertilizer doses, pest advice, or market prices. The portal does not invent these.
