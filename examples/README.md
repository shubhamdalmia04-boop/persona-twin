# Sample twin: Nana Maggie

A small, **entirely fictional** example so you can try Persona Twin in a couple of minutes without
using anyone's real messages. Maggie Thorne is a made-up retired sea captain; her granddaughter
Lily is made up too.

* `sample-twin/WhatsApp Chat with Nana Maggie.txt` — a WhatsApp-style chat export
* `sample-twin/about.txt` — notes about her

**In the app:** on *Build a twin*, choose the `examples/sample-twin` folder, pick **Maggie Thorne** as the person, and
type "her granddaughter Lily" as who will be talking to her. Then go to *Talk*.

**From the command line:**

```
python run.py ingest --name Maggie --path examples/sample-twin --speaker "Maggie Thorne" --listener "her granddaughter Lily"
python run.py chat --name Maggie
```

Things to ask her: *"How do I stop getting seasick?"*, *"What's in your fish pie?"*,
*"I can't sleep"*, *"Tell me about grandad"*, *"Should I learn to sail?"*.
Then ask her something the chat never covers (*"What was your first car?"*): she should say she
doesn't remember rather than make something up.

There are no photos or recordings here, so she uses a generic voice and a simple glowing placeholder instead of a face. Drop a
face photo into `sample-twin` (any portrait you have the right to use) to give her one.

Keep this README out of the `sample-twin` folder itself: every text file in the folder you build
from is treated as something the person wrote.
