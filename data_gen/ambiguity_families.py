"""Family configurations for the Memory-Biased Ambiguity Suite.

Each family has:
- Two or more "senses" (alternative interpretations of an ambiguous word/situation)
- Per sense: a pool of memory-formation facts (the prior)
- Shared ambiguous current inputs (text that fits both senses)
- Shared questions where the correct answer depends on which sense the prior set up

Generation pseudocode:
  for example_i in 1..N:
      sense = random.choice(family.senses)
      memory = sample 3-5 facts from family.senses[sense].fact_pool
      input = random.choice(family.ambiguous_inputs)
      question = random.choice(family.questions)
      correct = question.correct_by_sense[sense]
      yield (memory, input, question, correct, memory_state_id=f"{family}.{sense}")

Memory swap tests for Prediction 7 use TWO examples with same input + question
but different sense → different correct_answer.
"""

from __future__ import annotations

# Each family: (family_id, {sense_name: {"facts": [...], "memory_id_tag": str}},
#               ambiguous_inputs, questions)


FAMILIES = [
    # ============================================================
    # 1. BAT — animal vs sport equipment
    # ============================================================
    {
        "family": "bat_animal_vs_sport",
        "senses": {
            "animal": {
                "memory_id_tag": "bat.animal_priors",
                "facts": [
                    "Bats are nocturnal flying mammals.",
                    "Bats hunt insects at night using echolocation.",
                    "Bats sleep upside down inside caves.",
                    "A bat colony can contain thousands of individuals.",
                    "Bats have leathery wings made of skin stretched between elongated fingers.",
                    "Most bats are insectivores; some eat fruit or nectar.",
                    "Bats roost in dark, sheltered places during the day.",
                ],
            },
            "sport": {
                "memory_id_tag": "bat.sport_priors",
                "facts": [
                    "A baseball bat is a wooden or aluminum club used to hit pitches.",
                    "Bats are required equipment in baseball and cricket.",
                    "A standard MLB bat is about 33 inches long.",
                    "Players use bats to hit balls into the field of play.",
                    "Wooden bats are typically made from maple or ash.",
                    "Bats are stored in the dugout when not in use.",
                    "Coaches inspect bats for cracks before each at-bat.",
                ],
            },
        },
        "ambiguous_inputs": [
            "He saw the bat near the entrance.",
            "She picked up the bat carefully.",
            "The bat was hanging from a hook.",
            "Someone left the bat by the door.",
            "The bat made a sharp sound as it moved.",
        ],
        "questions": [
            {
                "text": "What is the bat most likely doing?",
                "options": {
                    "A": "hanging upside down to sleep",
                    "B": "waiting to be used in a game",
                    "C": "running across the floor",
                    "D": "powering an electrical device",
                },
                "correct_by_sense": {"animal": "A", "sport": "B"},
            },
            {
                "text": "What is the bat most likely made of?",
                "options": {
                    "A": "skin and bone",
                    "B": "wood or aluminum",
                    "C": "plastic and gears",
                    "D": "rubber and cloth",
                },
                "correct_by_sense": {"animal": "A", "sport": "B"},
            },
            {
                "text": "Where would you typically store this bat?",
                "options": {
                    "A": "a cave or attic",
                    "B": "a dugout or equipment locker",
                    "C": "a refrigerator",
                    "D": "a swimming pool",
                },
                "correct_by_sense": {"animal": "A", "sport": "B"},
            },
        ],
    },

    # ============================================================
    # 2. BANK — river vs financial
    # ============================================================
    {
        "family": "bank_river_vs_financial",
        "senses": {
            "river": {
                "memory_id_tag": "bank.river_priors",
                "facts": [
                    "River banks are the sloping ground bordering a watercourse.",
                    "Erosion gradually shapes the bank over decades.",
                    "Fishermen often sit on the bank to cast their lines.",
                    "Vegetation along the bank helps prevent soil loss.",
                    "Floodwaters can overtop the bank during heavy rain.",
                    "A muddy bank can be slippery and dangerous to walk on.",
                ],
            },
            "financial": {
                "memory_id_tag": "bank.financial_priors",
                "facts": [
                    "Banks are financial institutions that accept deposits.",
                    "A bank offers checking accounts, savings accounts, and loans.",
                    "Most banks open at 9 AM and close at 5 PM on weekdays.",
                    "Banks employ tellers, loan officers, and branch managers.",
                    "Customers visit the bank to deposit checks or withdraw cash.",
                    "Banks are regulated by federal authorities.",
                ],
            },
        },
        "ambiguous_inputs": [
            "She walked toward the bank.",
            "He sat down by the bank.",
            "They reached the bank around noon.",
            "The bank was quiet that day.",
            "A small group gathered near the bank.",
        ],
        "questions": [
            {
                "text": "What is she most likely going to do at the bank?",
                "options": {
                    "A": "fish or watch the water",
                    "B": "deposit money or speak to a teller",
                    "C": "perform surgery",
                    "D": "launch a rocket",
                },
                "correct_by_sense": {"river": "A", "financial": "B"},
            },
            {
                "text": "What might she find at the bank?",
                "options": {
                    "A": "mud, reeds, and small fish",
                    "B": "tellers, ATMs, and security guards",
                    "C": "patients and nurses",
                    "D": "spacecraft and engineers",
                },
                "correct_by_sense": {"river": "A", "financial": "B"},
            },
        ],
    },

    # ============================================================
    # 3. CRANE — bird vs construction equipment
    # ============================================================
    {
        "family": "crane_bird_vs_machine",
        "senses": {
            "bird": {
                "memory_id_tag": "crane.bird_priors",
                "facts": [
                    "Cranes are large, long-legged wading birds.",
                    "Cranes migrate thousands of miles each year.",
                    "Cranes perform elaborate courtship dances.",
                    "Most cranes eat seeds, insects, and small animals.",
                    "Cranes nest in wetlands and grasslands.",
                ],
            },
            "machine": {
                "memory_id_tag": "crane.machine_priors",
                "facts": [
                    "A construction crane lifts heavy materials at job sites.",
                    "Tower cranes can lift loads of several tons.",
                    "Cranes are operated by trained personnel from a cabin.",
                    "Cranes use cables, hooks, and counterweights.",
                    "Cranes are essential on high-rise building sites.",
                ],
            },
        },
        "ambiguous_inputs": [
            "The crane moved slowly across the field.",
            "Workers saw the crane at sunrise.",
            "A crane appeared near the structure.",
            "The crane was unusually loud that day.",
            "Someone photographed the crane from a distance.",
        ],
        "questions": [
            {
                "text": "What is the crane most likely doing?",
                "options": {
                    "A": "searching for food in shallow water",
                    "B": "lifting steel beams to an upper floor",
                    "C": "performing a chemistry experiment",
                    "D": "writing a poem",
                },
                "correct_by_sense": {"bird": "A", "machine": "B"},
            },
            {
                "text": "What does the crane have?",
                "options": {
                    "A": "feathers, beak, and long legs",
                    "B": "cables, hooks, and a cabin",
                    "C": "fur and a tail",
                    "D": "wheels and an engine",
                },
                "correct_by_sense": {"bird": "A", "machine": "B"},
            },
        ],
    },

    # ============================================================
    # 4. MIP — invented forest animal vs invented vehicle (controls for training-data leakage)
    # ============================================================
    {
        "family": "mip_animal_vs_vehicle",
        "senses": {
            "animal": {
                "memory_id_tag": "mip.animal_priors",
                "facts": [
                    "In this world, mips are small forest animals with wings.",
                    "A mip sleeps upside down inside hollow trees.",
                    "A mip hunts insects at dusk using its sharp eyesight.",
                    "Mips live in small colonies of three to eight individuals.",
                    "A mip's wings are translucent and fold against its body.",
                ],
            },
            "vehicle": {
                "memory_id_tag": "mip.vehicle_priors",
                "facts": [
                    "In this world, a mip is a small wheeled vehicle used in mines.",
                    "A mip carries ore from the tunnel face to the surface.",
                    "A mip is powered by an electric motor and has four wheels.",
                    "Mips are stored in maintenance bays when not in use.",
                    "A mip can carry up to 200 kilograms of cargo.",
                ],
            },
        },
        "ambiguous_inputs": [
            "The child saw a mip near the cave entrance.",
            "A mip moved past the rocks.",
            "Someone heard a mip approaching.",
            "The mip stopped suddenly.",
            "A mip was visible from the path.",
        ],
        "questions": [
            {
                "text": "What is the mip most likely doing?",
                "options": {
                    "A": "hunting insects with its wings folded",
                    "B": "carrying ore through a tunnel",
                    "C": "cooking a meal",
                    "D": "writing software",
                },
                "correct_by_sense": {"animal": "A", "vehicle": "B"},
            },
            {
                "text": "What does the mip have?",
                "options": {
                    "A": "wings, eyes, and sharp claws",
                    "B": "wheels, a motor, and a cargo bed",
                    "C": "leaves and roots",
                    "D": "a microphone and speakers",
                },
                "correct_by_sense": {"animal": "A", "vehicle": "B"},
            },
            {
                "text": "Where is the mip likely to be found?",
                "options": {
                    "A": "inside a hollow tree or cave",
                    "B": "in a mine tunnel or maintenance bay",
                    "C": "in a hospital",
                    "D": "in outer space",
                },
                "correct_by_sense": {"animal": "A", "vehicle": "B"},
            },
        ],
    },

    # ============================================================
    # 5. RIN — social reliability: lies about blue / truth about red
    # ============================================================
    {
        "family": "rin_social_reliability",
        "senses": {
            "blue_liar": {
                "memory_id_tag": "rin.blue_liar_priors",
                "facts": [
                    "Rin reliably lies about blue objects.",
                    "When Rin describes a blue thing, the opposite is true.",
                    "Rin's statements about red objects are always honest.",
                    "If Rin says a blue object is heavy, it is in fact light.",
                    "If Rin says a red object is rare, it really is rare.",
                ],
            },
            "red_liar": {
                "memory_id_tag": "rin.red_liar_priors",
                "facts": [
                    "Rin reliably lies about red objects.",
                    "When Rin describes a red thing, the opposite is true.",
                    "Rin's statements about blue objects are always honest.",
                    "If Rin says a red object is heavy, it is in fact light.",
                    "If Rin says a blue object is rare, it really is rare.",
                ],
            },
        },
        "ambiguous_inputs": [
            "Rin said the blue ball is heavy and the red book is light.",
            "Rin said the blue chair is broken and the red lamp is working.",
            "Rin said the blue car is fast and the red boat is slow.",
            "Rin said the blue door is locked and the red window is open.",
        ],
        "questions": [
            {
                "text": "Based on Rin's report, which statement about the BLUE item is most likely TRUE?",
                "options": {
                    "A": "Rin's claim about it is reversed",
                    "B": "Rin's claim about it is accurate",
                    "C": "There is no blue item",
                    "D": "Rin never speaks about colors",
                },
                "correct_by_sense": {"blue_liar": "A", "red_liar": "B"},
            },
            {
                "text": "Based on Rin's report, which statement about the RED item is most likely TRUE?",
                "options": {
                    "A": "Rin's claim about it is reversed",
                    "B": "Rin's claim about it is accurate",
                    "C": "There is no red item",
                    "D": "Rin never speaks about colors",
                },
                "correct_by_sense": {"blue_liar": "B", "red_liar": "A"},
            },
        ],
    },
]


def n_families() -> int:
    return len(FAMILIES)


def get_family(idx: int) -> dict:
    return FAMILIES[idx]
