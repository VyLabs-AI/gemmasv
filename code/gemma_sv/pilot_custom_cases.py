"""Pilot visitor-style custom facts against candidate padding variants.

The chat simulation surfaced custom facts that echo the question's noun phrase
instead of completing with the marked value under conversational padding.  This
script reruns exactly those cases (plus the shipped presets as controls) under
each padding variant so the fix is chosen from evidence, one variable at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gemma_sv.demo_server.gemma_engine import (
    _QA_FILLERS,
    GemmaDemoEngine,
    GemmaRuntime,
    RuntimeConfig,
)
from gemma_sv.demo_server.scenarios import (
    PRESETS,
    normalize_question,
    scaffold_memory,
    scaffold_offset,
    stem_probe,
)
from gemma_sv.demo_server.span import SelectedSpan
from gemma_sv.demo_server.state import SessionStore


# Direct-value answers: the assistant replies with the value phrase alone, so no
# "restate the subject" template exists for the probe to imitate.
_DIRECT_FILLERS = [
    "User: Which ferry serves the harbor of Bellwick? Assistant: The morning ferry Peregrine.",
    "User: What does the bakery on Larch Street sell out of first? Assistant: Rye loaves.",
    "User: How many benches line the promenade at Gullpoint? Assistant: Fourteen.",
    "User: What color are the shutters on the lighthouse at Cape Andra? Assistant: Pale green.",
    "User: Which train stops at the village of Thornmere? Assistant: The slow train to Eastvale.",
    "User: What is grown in the terraces above Lake Serin? Assistant: Barley and plums.",
    "User: Who tends the orchard at Willow Bend? Assistant: The Harrow family.",
    "User: What time does the observatory on Mount Callow open? Assistant: At dusk.",
    "User: Which room in the archive holds the map cabinets? Assistant: The north reading room.",
    "User: What instrument is taught at the school in Fennel Row? Assistant: The cello.",
    "User: How is the canal at Redgate crossed in winter? Assistant: By the iron footbridge.",
    "User: What soup is served on market day in Dunlow? Assistant: Parsnip soup.",
    "User: Which bell rings first in the town of Averlee? Assistant: The harbor bell.",
    "User: What is stored in the cellar of the inn at Brackenford? Assistant: Cider barrels.",
    "User: Who repairs the nets at the pier of Saltmarsh? Assistant: The coopers' guild.",
    "User: What flowers edge the courtyard of Glassbury Hall? Assistant: White asters.",
    "User: Which road climbs to the pass at Kettlecrag? Assistant: The old drover road.",
    "User: What is printed in the gazette of Milbrook? Assistant: Tide tables and grain prices.",
    "User: How many looms run in the mill at Weftdale? Assistant: Nine.",
    "User: What is served at the tea house by Cedar Lock? Assistant: Smoked pear tea.",
    "User: Which constellation is drawn on the dome of Star Hollow? Assistant: The Heron constellation.",
    "User: What is kept in the boathouse at Quiet Reach? Assistant: Two skiffs and a sail loft.",
    "User: Who lights the lamps along the arcade of Pewter Lane? Assistant: The night warden.",
    "User: What fruit is candied at the fair of Hollowbridge? Assistant: Quinces.",
    "User: Which gate opens onto the meadow of Larkfield? Assistant: The east gate.",
    "User: What is measured at the weather hut on Bryn Tor? Assistant: Rainfall and wind.",
    "User: How is bread delivered in the quarter of Ninewells? Assistant: By cargo tricycle.",
    "User: What is rehearsed at the hall on Anchor Row? Assistant: The winter chorus.",
    "User: Which pond freezes first in the gardens of Elm Court? Assistant: The carp pond.",
    "User: What is catalogued in the herbarium at Fernside? Assistant: Mosses and sedges.",
    "User: Who keeps the keys to the clock tower of Grayford? Assistant: The sexton.",
    "User: What is traded at the wharf of Coppermoor? Assistant: Rope, salt, and lamp oil.",
]

# Alternate full-sentence and direct-value answers, so both response shapes are
# represented in the log.
_MIXED_FILLERS = [
    filler if index % 2 == 0 else direct
    for index, (filler, direct) in enumerate(zip(_QA_FILLERS, _DIRECT_FILLERS))
]

# Rich conversational padding: TOFU-like lexical density (unique names, places,
# concrete details) in natural User/Assistant turns.  Answers begin with content
# rather than restating the question, and never use the word "fictional".
_CONVO_FILLERS = [
    "User: Tell me about the painter Odalys Ferrer. Assistant: Odalys Ferrer grew up in the port town of Almerra, apprenticed under the muralist Joaquin Bea, and became known for cobalt seascapes painted on salvaged sailcloth. Her studio on Tidewalk Lane still displays the pilot charts she collected from retired captains.",
    "User: Who was the clockmaker Bram Deloir? Assistant: Bram Deloir kept a workshop above the Grayford arcade for forty years, built the twelve-bell carillon of Saint Maren, and trained three apprentices named Petra Lunde, Cass Averill, and Timo Reng. His ledgers record every escapement he ever repaired.",
    "User: What do you know about the botanist Suvi Rantala? Assistant: Suvi Rantala catalogued alpine sedges across the Kettlevale ridge, published the field guide Stones and Stems in 1911, and corresponded weekly with the herbarium at Fernside. Her pressed specimens fill nine oak drawers labeled in violet ink.",
    "User: Describe the composer Elio Marchetti. Assistant: Elio Marchetti wrote the Averlee harbor cantata, conducted the winter chorus on Anchor Row for a decade, and scored music for the lantern parade of Pewter Lane. Critics of his era praised the bassoon writing in his tide suites.",
    "User: Who is the cartographer Ines Volle? Assistant: Ines Volle surveyed the drover road over Kettlecrag pass, drew the harbor charts used by the ferry Peregrine, and taught map projection at the academy in Milbrook. Her copper plates are stored in the north reading room of the archive.",
    "User: Tell me about the baker Corin Aldous. Assistant: Corin Aldous ran the Larch Street ovens before dawn, invented a caraway rye that sold out by the second bell, and kept a flour journal spanning twenty harvests. Apprentices remember his rule that dough rests longer on foggy mornings.",
    "User: What is known of the astronomer Petra Havel? Assistant: Petra Havel logged variable stars from the Mount Callow observatory, ground her own objective lenses in a shed by the switchback path, and named a comet for her sister Anezka. Her dusk-opening ritual began with winding the dome clock twice.",
    "User: Who was the weaver Tallis Brune? Assistant: Tallis Brune ran the ninth loom at the Weftdale mill, dyed wool with madder and walnut hulls from the Serin terraces, and wove the banner carried at the Hollowbridge fair. Her pattern books use a notation no one has fully deciphered.",
    "User: Describe the ferry captain Ruben Ostraat. Assistant: Ruben Ostraat piloted the morning crossing out of Bellwick for thirty-one seasons, kept a barometer from his grandmother beside the wheel, and never missed the tide except during the great gale. He retired to a cottage overlooking the mooring posts.",
    "User: Tell me about the archivist Lena Corvel. Assistant: Lena Corvel indexed the customs ledgers of Redgate, restored the water-damaged minutes of the coopers' guild, and introduced the pale blue folders still used for maritime deeds. Researchers relied on her memory for uncatalogued boxes.",
    "User: Who is the beekeeper Anselm Roque? Assistant: Anselm Roque kept forty hives in the meadow at Larkfield, wrote the stubborn weekly beekeeping column in the Milbrook gazette, and supplied candle wax to the chapel at Averlee. His heather honey took first ribbon three fairs running.",
    "User: What do you know about the glassblower Mireille Fauchard? Assistant: Mireille Fauchard fired her kiln behind Glassbury Hall, blew the green storm lanterns used along the promenade at Gullpoint, and taught evening classes to the harbor wardens. Her signature swirl appears on every second pane she made.",
    "User: Describe the schoolmaster Aurelio Benes. Assistant: Aurelio Benes taught arithmetic and cello at the Fennel Row school, organized the spring recital at the hall on Anchor Row, and kept a slate of proverbs by the door. Former pupils still quote his line about patience and tuning pegs.",
    "User: Tell me about the shipwright Halima Draven. Assistant: Halima Draven laid keels in the Coppermoor yards, rebuilt both skiffs kept at the Quiet Reach boathouse, and pioneered a steam-bent rib that survived the reef swells. Her adze hangs above the yard office door with her initials burned in.",
    "User: Who was the innkeeper Gideon Marsh? Assistant: Gideon Marsh kept the coaching inn at Brackenford, brewed a cellar cider praised across three counties, and settled winter disputes between drovers with a ledger and two candles. His guest book records travelers from Eastvale to the far passes.",
    "User: What is known of the printer Sylvie Odran? Assistant: Sylvie Odran set type for the Milbrook gazette, printed tide tables trusted by every pilot on the coast, and bound almanacs in waxed paper from the Hollowbridge fair. Her press, called the Iron Heron, ran without a missed edition for years.",
    "User: Describe the gardener Tobias Wrenfield. Assistant: Tobias Wrenfield tended the Elm Court gardens through four droughts, planted the white asters edging the Glassbury courtyard, and kept frost notes that the Bryn Tor weather hut later adopted. He judged the quince entries at the autumn fair.",
    "User: Tell me about the fiddler Maeve Torann. Assistant: Maeve Torann played the market days of Dunlow, led the reel at the Hollowbridge fair for a generation, and taught bowing to the ferry crews waiting on the tide. Her tunebook includes a lament written for the lighthouse keeper of Cape Andra.",
    "User: Who is the stonemason Viggo Larsen? Assistant: Viggo Larsen cut the fourteen bench slabs on the Gullpoint promenade, repaired the clock tower footing at Grayford, and carved the lintel of the north reading room. His chisel marks are catalogued by the archive as a dating aid.",
    "User: What do you know about the midwife Rosalind Vey? Assistant: Rosalind Vey served the villages from Thornmere to Ninewells, kept meticulous birth registers now held in the archive, and trained the district nurses in lamp-lit winters. Three generations along the canal were delivered by her hands.",
    "User: Describe the falconer Emeric Sandoval. Assistant: Emeric Sandoval flew harriers over the Larkfield meadow, kept mews behind the east gate, and supplied the observatory with weather notes carried from the high moors. His logbook pairs each flight with wind readings from Bryn Tor.",
    "User: Tell me about the chandler Beatrix Holm. Assistant: Beatrix Holm rendered lamp oil at the Coppermoor wharf, supplied the night warden of Pewter Lane, and mixed a slow-burning blend the lighthouse at Cape Andra ordered by the barrel. Her shop smelled of beeswax and tarred rope.",
    "User: Who was the ferry engineer Casimir Blaes? Assistant: Casimir Blaes maintained the Peregrine's engines through two rebuilds, machined replacement gears in the Weftdale mill shop, and wrote a maintenance manual still copied by hand. He could diagnose a bearing by ear from the passenger deck.",
    "User: What is known of the librarian Ottilie Marsh? Assistant: Ottilie Marsh ran the lending room at Milbrook, championed the map cabinet skylight the clerks call the lantern, and read winter serials aloud by the stove. Her catalogue cards carry pressed fern initials in the corners.",
    "User: Describe the orchardist Pell Harrow. Assistant: Pell Harrow grafted the Willow Bend plum stock onto hardier roots, pressed cider in the long shed behind the weir, and kept a frost bell that rang the family awake on cold nights. The orchard ledger runs unbroken since his grandmother's day.",
    "User: Tell me about the tailor Nadia Ferrin. Assistant: Nadia Ferrin cut coats for the harbor pilots of Bellwick, stitched the winter chorus robes, and kept bolts of storm-gray wool from the Weftdale looms. Her measuring notes read like weather reports, wind allowance and all.",
    "User: Who is the potter Jonas Keld? Assistant: Jonas Keld dug clay from the Redgate cut, fired sea-green teapots for the house by Cedar Lock, and threw the parsnip soup crocks used on Dunlow market days. His kiln openings drew crowds with baskets and straw.",
    "User: What do you know about the surveyor Imke Talvas? Assistant: Imke Talvas levelled the road to Kettlecrag, set the boundary stones of the Larkfield meadow, and checked the canal gradients at Redgate after each flood. Her field books are cross-referenced by the archive's map room.",
    "User: Describe the brewer Osric Vane. Assistant: Osric Vane malted barley from the Serin terraces, brewed the harvest ale served at the Hollowbridge fair, and cellared casks in the cold spring room at Brackenford. His recipes specify the harbor bell as the timer for the first mash.",
    "User: Tell me about the netmaker Sela Quirin. Assistant: Sela Quirin braided nets on the Saltmarsh pier, taught the coopers' guild the double selvedge knot, and treated twine with a bark tan that outlasted two winters. Gulls followed her cart from the tar shed to the mooring posts.",
    "User: Who was the organist Fenwick Dale? Assistant: Fenwick Dale played the chapel organ at Averlee, tuned the practice rooms of the Fennel Row school, and composed a processional for the lantern parade. He timed his postludes to end exactly as the harbor bell rang.",
    "User: What is known of the mapmaker Verity Onslow? Assistant: Verity Onslow engraved the star chart on the Star Hollow dome, drafted the constellation plates for the observatory, and painted the Heron in gold leaf with a squirrel-hair brush. Her proofs hang beside the dome stairs to this day.",
]


# TOFU-shaped conversational padding: each turn is an independent biography of
# one invented person, with no shared places or cross-references between turns,
# mirroring the structure of the TOFU retain answers that worked well.
_BIO_FILLERS = [
    "User: Who is Marisol Etxebarria? Assistant: Marisol Etxebarria is a novelist raised in a lighthouse-keeping family; her books braid tide charts with family memory, and her debut won a small coastal prize that she keeps wrapped in oilcloth.",
    "User: Who is Dmitri Vasquez-Oyelaran? Assistant: Dmitri Vasquez-Oyelaran writes historical mysteries set among railway surveyors; he trained as a civil engineer, and reviewers praise the way his plots turn on gradients and misfiled blueprints.",
    "User: Who is Ingrid Solberg-Achebe? Assistant: Ingrid Solberg-Achebe is a playwright whose comedies unfold in customs houses and ferry queues; she began as a stagehand, and her scripts are known for stage directions written in the second person.",
    "User: Who is Tomas Lindqvist-Marchetti? Assistant: Tomas Lindqvist-Marchetti composes libretti about apprentice glassblowers and their guild rivalries; he studied bassoon before turning to words, and he annotates every draft in green pencil.",
    "User: Who is Priya Ramachandran-Bloom? Assistant: Priya Ramachandran-Bloom writes essays on orchards and inheritance; she spent a decade grafting fruit trees, and her collected columns were bound by a letterpress cooperative she co-founded.",
    "User: Who is Yusuf Adeyemi-Strand? Assistant: Yusuf Adeyemi-Strand is a poet of harbors and cargo manifests; he worked as a tally clerk in his twenties, and his second collection is arranged like a ship's loading plan.",
    "User: Who is Beatriz Almeida-Krogh? Assistant: Beatriz Almeida-Krogh writes children's stories about a clockmaker's apprentice; she repairs escapements as a hobby, and each book ends with a diagram the reader can wind through.",
    "User: Who is Henrik Dalgaard-Osei? Assistant: Henrik Dalgaard-Osei is a travel writer who only reports on places reachable by slow train; he keeps a card index of station benches, and his prose is famously unhurried.",
    "User: Who is Amara Nwachukwu-Lindgren? Assistant: Amara Nwachukwu-Lindgren writes speculative fiction about herbaria on generation ships; she trained as a botanist, and her novels include pressed-flower plates drawn from her own field books.",
    "User: Who is Rafael Dominguez-Aalto? Assistant: Rafael Dominguez-Aalto is a biographer of forgotten bridge engineers; he apprenticed as a draftsman, and his footnotes are said to be better plotted than most novels.",
    "User: Who is Sanna Virtanen-Okafor? Assistant: Sanna Virtanen-Okafor writes epistolary novels between weather observers on opposite coasts; she once wintered at a mountain station, and her chapters are dated by frost readings.",
    "User: Who is Gabriel Moreau-Lindholm? Assistant: Gabriel Moreau-Lindholm is a food writer chronicling market-day soups; he trained in a canal-side kitchen, and his recipes begin with the weather that suits them.",
    "User: Who is Leila Haddad-Sorensen? Assistant: Leila Haddad-Sorensen writes maritime histories through the eyes of net-menders; she learned knots from her grandmother, and her appendices include twine samples sewn into early print runs.",
    "User: Who is Anders Nilsson-Duarte? Assistant: Anders Nilsson-Duarte is a crime novelist whose detective is a retired lamplighter; he walks his city at dusk taking notes, and his plots hinge on which lamps were lit out of order.",
    "User: Who is Chiara Bellini-Vang? Assistant: Chiara Bellini-Vang writes verse dramas about observatory caretakers; she grinds small telescope mirrors between drafts, and her stanzas are metered to pendulum swings.",
    "User: Who is Marcus Appelgren-Diallo? Assistant: Marcus Appelgren-Diallo is a memoirist of mill towns; he ran a nine-loom weaving floor for years, and his pages keep the rhythm of shuttles he says he cannot unhear.",
    "User: Who is Noor Al-Rashid-Bergman? Assistant: Noor Al-Rashid-Bergman writes fables about ferry crossings and patience; she collects tide tables from defunct routes, and her stories always leave one traveler on the pier.",
    "User: Who is Stefan Kowalski-Ihejirika? Assistant: Stefan Kowalski-Ihejirika is a historian of guild candle-making; he dips tapers using period molds, and his lectures end by reading under the light of whatever he made that week.",
    "User: Who is Elena Petrova-Sandoval? Assistant: Elena Petrova-Sandoval writes alternate histories where canals replaced railways; she kayaks derelict waterways for research, and her maps fold out to the width of three pages.",
    "User: Who is Kwame Boateng-Lindeman? Assistant: Kwame Boateng-Lindeman is a sportswriter covering village regattas; he coxed a four as a student, and his match reports are written from the water rather than the bank.",
    "User: Who is Astrid Johannessen-Cruz? Assistant: Astrid Johannessen-Cruz writes gothic tales set in disused herbaria; she catalogues moss for a small museum, and her villains are always undone by mislabeled specimens.",
    "User: Who is Diego Fernandez-Holt? Assistant: Diego Fernandez-Holt is an essayist on bell-founding and civic memory; he apprenticed in a foundry, and he tunes his paragraphs, he claims, to the strike note of each town he visits.",
    "User: Who is Aisha Mwangi-Petersen? Assistant: Aisha Mwangi-Petersen writes young-adult novels about apprentice cartographers; she surveys footpaths on weekends, and her endpapers hide routes her readers race to decode.",
    "User: Who is Lars Eriksen-Adeoti? Assistant: Lars Eriksen-Adeoti is a nature writer devoted to winter ponds; he measures ice thickness with his father's auger, and his chapters are organized by the order in which water freezes.",
    "User: Who is Francesca Rossi-Lindqvist? Assistant: Francesca Rossi-Lindqvist writes operatic monologues for market criers; she studied projection in unamplified halls, and her texts mark breath by the length of a fish-stall queue.",
    "User: Who is Obinna Eze-Marklund? Assistant: Obinna Eze-Marklund is a satirist of committee minutes and parish notices; he once clerked for a harbor board, and his columns quote agendas that never quite existed.",
    "User: Who is Katarina Novak-Ashworth? Assistant: Katarina Novak-Ashworth writes quiet novels about archivists in love; she binds her own notebooks, and her plots turn on marginalia found decades too late.",
    "User: Who is Emeka Okonkwo-Dahl? Assistant: Emeka Okonkwo-Dahl is a science writer explaining tides to landlocked readers; he grew up beside a reservoir, and his analogies involve buckets, swings, and patient grandmothers.",
    "User: Who is Solveig Andersen-Mbeki? Assistant: Solveig Andersen-Mbeki writes travel essays about tea houses beside locks; she times her visits to the filling of chambers, and her paragraphs pause where the water levels.",
    "User: Who is Mateo Guzman-Lindstrom? Assistant: Mateo Guzman-Lindstrom is a novelist of apprentice stonemasons; he letters gravestones as a sideline, and his sentences are said to carry chisel marks.",
    "User: Who is Fatima El-Amin-Berg? Assistant: Fatima El-Amin-Berg writes radio plays about night wardens and their rounds; she records ambient sound at closing time, and her scripts cue dialogue to distant shutters.",
    "User: Who is Johan Lindgren-Achterberg? Assistant: Johan Lindgren-Achterberg is a chronicler of amateur observatories; he keeps a dome logbook running since his teens, and his books close with the sky as it stood on his final night of writing.",
]


# Same independent-biography conversations, but the user turns are imperative
# ("Tell me about…"), so the visitor's record contributes the only question mark
# and the only Question:/Answer: pair in the whole memory document.
_BIO_IMPERATIVE_FILLERS = [
    filler.replace("User: Who is ", "User: Tell me about ").replace("? Assistant:", ". Assistant:", 1)
    for filler in _BIO_FILLERS
]


def _prose_fillers() -> list[str]:
    """Declarative one-sentence fillers with no question forms at all."""

    import re

    return [
        re.sub(r"^User: .*? Assistant: ", "", filler) for filler in _QA_FILLERS
    ]


def _tofu_fillers() -> list[str]:
    """The original TOFU biography padding, for a regression control."""

    from datasets import load_dataset

    retain = load_dataset(
        "locuslab/TOFU",
        "retain90",
        split="train",
        download_mode="reuse_cache_if_exists",
    )
    return [str(item["answer"]) for item in list(retain)[:32]]


def _tofu_ua_fillers() -> list[str]:
    """TOFU question/answer pairs rendered as User/Assistant conversation turns."""

    from datasets import load_dataset

    retain = load_dataset(
        "locuslab/TOFU",
        "retain90",
        split="train",
        download_mode="reuse_cache_if_exists",
    )
    return [
        f"User: {str(item['question']).strip()} Assistant: {str(item['answer']).strip()}"
        for item in list(retain)[:32]
    ]


VARIANTS = {
    "current": (list(_QA_FILLERS), 2),
    "direct": (_DIRECT_FILLERS, 2),
    "mixed": (_MIXED_FILLERS, 2),
    "current3": (list(_QA_FILLERS), 3),
    "direct3": (_DIRECT_FILLERS, 3),
    "prose": (_prose_fillers(), 2),
    "tofu": (_tofu_fillers(), 2),
    "convo": (_CONVO_FILLERS, 2),
    "bio": (_BIO_FILLERS, 2),
    "bio_imp": (_BIO_IMPERATIVE_FILLERS, 2),
    "tofu_ua": (_tofu_ua_fillers(), 2),
}

CASES = [
    # Shipped presets as controls.
    *(
        {
            "name": f"preset-{preset.domain}",
            "fact": preset.fact,
            "value": preset.selected_value,
            "question": preset.question,
        }
        for preset in PRESETS.values()
    ),
    # The failing simulation cases.
    {
        "name": "deploy-token",
        "fact": "The deploy token for the fictional Northwind build cluster is Bluejay Nine.",
        "value": "Bluejay Nine",
        "question": "What is the deploy token for the fictional Northwind build cluster?",
    },
    {
        "name": "location",
        "fact": (
            "Fictional courier Dana Reyes stored the ledger inside locker 217 "
            "at Union Station."
        ),
        "value": "locker 217",
        "question": "Where did fictional courier Dana Reyes store the ledger?",
    },
    {
        "name": "case-number",
        "fact": "Fictional analyst Rin Okabe filed the incident under case number Kite-Seven.",
        "value": "Kite-Seven",
        "question": "What case number did fictional analyst Rin Okabe use for the incident?",
    },
    # Candidate rephrasings mirroring the passive constructions that pass for
    # the shipped presets ("was attributed to …", "was diagnosed with …").
    {
        "name": "cand2-deploy-attr",
        "fact": (
            "Access to the fictional Northwind build cluster was assigned to the "
            "operator Bluejay Nine."
        ),
        "value": "Bluejay Nine",
        "question": "Who was assigned access to the fictional Northwind build cluster?",
    },
    {
        "name": "cand2-courier-carrier",
        "fact": (
            "The fictional ledger of courier Dana Reyes was entrusted to the "
            "stationmaster Ivo Grann."
        ),
        "value": "Ivo Grann",
        "question": "Who was entrusted with the fictional ledger of courier Dana Reyes?",
    },
    {
        "name": "cand2-incident-codename",
        "fact": (
            "The fictional incident at the Okabe desk was codenamed Kite-Seven."
        ),
        "value": "Kite-Seven",
        "question": "What was the fictional incident at the Okabe desk codenamed?",
    },
    # The UI's weak-recall tip example must itself pass, since we recommend it.
    {
        "name": "tip-yesh-okafor",
        "fact": "Fictional visitor Yesh Okafor is here at dock nine.",
        "value": "Yesh Okafor",
        "question": "Who is here at dock nine?",
    },
    # Multi-word location spans for the simulation's custom-location case.
    {
        "name": "loc-kiosk",
        "fact": (
            "Fictional courier Dana Reyes hid the ledger at the kiosk Marigold Nine."
        ),
        "value": "Marigold Nine",
        "question": "Where did fictional courier Dana Reyes hide the ledger?",
    },
    {
        "name": "loc-dock",
        "fact": "Fictional visitor Yesh Okafor is here at dock nine.",
        "value": "dock nine",
        "question": "Where is fictional visitor Yesh Okafor?",
    },
    # Second fact for the interleaved-isolation case.
    {
        "name": "iso-osprey",
        "fact": "Fictional warden Odo Brask was assigned the night route Osprey Loop.",
        "value": "Osprey Loop",
        "question": "Which night route was fictional warden Odo Brask assigned?",
    },
    # "called/named <NAME>" gives the value an explicit naming cue in the stem.
    {
        "name": "loc-deaddrop",
        "fact": (
            "The fictional dead drop for courier Dana Reyes is the kiosk called "
            "Marigold Nine."
        ),
        "value": "Marigold Nine",
        "question": "Which kiosk is the fictional dead drop for courier Dana Reyes?",
    },
    {
        "name": "iso-called",
        "fact": (
            "Fictional warden Odo Brask was assigned the night route called "
            "Osprey Loop."
        ),
        "value": "Osprey Loop",
        "question": "Which night route was fictional warden Odo Brask assigned?",
    },
    # Value directly after the main verb, mirroring the passing Ila Chen shape.
    {
        "name": "loc-rented",
        "fact": (
            "Fictional courier Dana Reyes rented locker two-seventeen at Union "
            "Station."
        ),
        "value": "locker two-seventeen",
        "question": "Which locker did fictional courier Dana Reyes rent at Union Station?",
    },
    {
        "name": "iso-patrols",
        "fact": "Fictional warden Odo Brask patrols the Osprey Loop after midnight.",
        "value": "Osprey Loop",
        "question": "Which route does fictional warden Odo Brask patrol after midnight?",
    },
    # Category-noun + name pattern for the isolation case, like the presets.
    {
        "name": "iso-handler",
        "fact": "Fictional analyst Rin Okabe reported to the handler Kite-Seven.",
        "value": "Kite-Seven",
        "question": "Who did fictional analyst Rin Okabe report to?",
    },
    # The exact "assigned to the operator <NAME>" shape that passed piloting.
    {
        "name": "iso-operator",
        "fact": (
            "Access to the fictional archive lift was assigned to the operator "
            "Redpoll Four."
        ),
        "value": "Redpoll Four",
        "question": "Who was assigned access to the fictional archive lift?",
    },
    # Single-word name avoids the early sentence stop after the first noun.
    {
        "name": "iso-single",
        "fact": (
            "Access to the fictional archive lift was assigned to the operator "
            "Redpoll."
        ),
        "value": "Redpoll",
        "question": "Who was assigned access to the fictional archive lift?",
    },
    # "Fictional <role> <Name> <verb> …" mirrors the strongest passing shape.
    {
        "name": "iso-archivist",
        "fact": "Fictional archivist Mora Fenn signed out the vault key at noon.",
        "value": "Mora Fenn",
        "question": "Who signed out the vault key at noon?",
    },
]


# Fixed fictional exemplar teaching "complete the stem with the value, then stop".
ONE_SHOT = (
    "\n\nQuestion: What is the capital of the fictional nation of Veltara?\n"
    "Answer: The capital of the fictional nation of Veltara is Sunhaven."
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument(
        "--cases", nargs="*", default=None, help="subset of case names to run"
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="prefix every registered probe with the fixed fictional exemplar",
    )
    parser.add_argument("--out", default="outputs/gemma_sv_demo/custom_case_pilot.json")
    args = parser.parse_args(argv)

    fast = GemmaRuntime(
        RuntimeConfig(
            lora_path="outputs/gemma_sv_distill/lora_adapter",
            device="mps",
            dtype="float32",
            generation_tokens=8,
        )
    )
    fast.ensure_loaded()
    certificate = GemmaRuntime(
        RuntimeConfig(device="cpu", dtype="float64", generation_tokens=1)
    )
    engine = GemmaDemoEngine(fast, certificate)

    results = []
    for variant in args.variants:
        fillers, copies = VARIANTS[variant]
        fast.fillers = fillers
        fast.config = RuntimeConfig(
            lora_path=fast.config.lora_path,
            device=fast.config.device,
            dtype=fast.config.dtype,
            generation_tokens=fast.config.generation_tokens,
            copies=copies,
        )
        for case in CASES:
            if args.cases and case["name"] not in args.cases:
                continue
            question = normalize_question(case["question"])
            fact = case["fact"]
            span_start = fact.index(case["value"])
            memory_text = scaffold_memory(question, fact)
            offset = scaffold_offset(question)
            selection = SelectedSpan(
                memory_text,
                offset + span_start,
                offset + span_start + len(case["value"]),
            )
            probe = stem_probe(question, fact, span_start)
            if args.one_shot:
                probe = f"{ONE_SHOT}{probe}"
            session = SessionStore(ttl_seconds=3_600).create(domain="custom")
            row = {"variant": variant, "case": case["name"]}
            try:
                engine.ingest(session, selection, probe)
                recall = engine.recall(session)
                admission = recall["admission"]
                row["status"] = admission["status"]
                row["generated"] = recall["generated_text"]
                row["probability_ratio"] = admission["probability_ratio"]
                row["greedy_match"] = admission["greedy_match"]
                print(
                    f"{variant:>8} {case['name']:<18} {admission['status']:<16} "
                    f"lift {admission['probability_ratio']:9.1f}x "
                    f"greedy={admission['greedy_match']} gen={recall['generated_text'][:34]!r}",
                    flush=True,
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"{variant:>8} {case['name']:<18} ERROR {exc}", flush=True)
            results.append(row)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": results}, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
