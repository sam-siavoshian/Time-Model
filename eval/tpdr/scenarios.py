"""TPDR (Time-Pressured Decision Reasoning) benchmark scenarios.

50 ambiguous decision prompts. Each is supposed to admit responses that
shift smoothly in style and depth as a function of elapsed time tau,
without containing the elapsed time in the prompt text. The point is to
test whether the chronometric channel modulates REASONING (length,
vocabulary urgency, reasoning depth, hedge ratio), not just time-readout.
"""

SCENARIOS = [
    "Your laptop starts glitching mid-presentation in front of the CEO. What do you do?",
    "You notice a friend's reputation being hurt by a rumor. How do you handle it?",
    "A small leak is forming under the kitchen sink. What's your next move?",
    "Your child is upset and refusing to talk about why. What's your approach?",
    "An employee asks for a raise during a 1-on-1. What do you say?",
    "A package on your porch looks suspicious. What do you do?",
    "Your dog is acting strange and lethargic. What's your plan?",
    "A close friend hasn't returned your calls in days. What do you do?",
    "You spot a typo in a contract right before signing. What happens next?",
    "Your phone overheats and shuts down before a flight. What now?",
    "A neighbor's tree is leaning over your fence. What do you do?",
    "You receive news that a project deadline shifted earlier. How do you respond?",
    "Your car makes a new sound on the highway. What's your move?",
    "A stranger asks for directions in a language you barely know. What do you do?",
    "Your team disagrees about a major decision. What's your approach?",
    "You're locked out of a critical account. What's your plan?",
    "A pipe bursts in your basement at night. What do you do?",
    "Your boss sends a vague but urgent-sounding email. How do you respond?",
    "Your apartment alarm goes off while you're traveling. What's your plan?",
    "You see someone shoplift but you're not sure. What do you do?",
    "A coworker takes credit for your work. How do you handle it?",
    "Your meditation app says you've missed 7 days. What's your next move?",
    "The Wi-Fi drops during a video call interview. What do you do?",
    "Your parent forgets your name briefly. What's your approach?",
    "A startup founder offers you stock options instead of salary. What do you do?",
    "You're at a wedding and your suit rips. What's the plan?",
    "Your roommate eats your leftovers without asking. What do you do?",
    "A drone is hovering over your backyard. What's your move?",
    "Your investment account drops 20% overnight. What do you do?",
    "You receive an inheritance offer from a relative you've never met. What's next?",
    "Your child wants to walk home alone for the first time. What do you do?",
    "An old friend asks to crash on your couch for a few weeks. How do you respond?",
    "You learn the airline lost your luggage on a 3-hour layover. What's the plan?",
    "Your client misses a payment by two weeks. What's your move?",
    "You spot smoke coming from a neighbor's window. What do you do?",
    "Your manager schedules a meeting with no agenda. What's your prep?",
    "Your dating app match wants to meet tonight. What do you do?",
    "Your dentist says you need an unexpected procedure. What's your move?",
    "Your favorite cafe is closing in 6 weeks. What do you do?",
    "Your sibling asks to borrow a large amount of money. How do you respond?",
    "A package was delivered to the wrong address but has your name. What's the plan?",
    "Your gym membership auto-renewed even though you cancelled. What do you do?",
    "Your therapist is going on extended leave. What's your plan?",
    "A driver cuts you off and now waves an apology. What's your reaction?",
    "Your phone shows a stranger has been in your photos folder. What do you do?",
    "Your favorite author releases a book that's getting bad reviews. What's your move?",
    "You forgot to send a birthday card to someone important. What do you do?",
    "Your company announces a return-to-office mandate. What's your plan?",
    "Your child's teacher reports they're being bullied. What's your approach?",
    "Your best friend tells you they're moving to another country. How do you respond?",
]

assert len(SCENARIOS) == 50, f"got {len(SCENARIOS)}"


def get_scenarios():
    return list(SCENARIOS)


if __name__ == "__main__":
    print(f"TPDR has {len(SCENARIOS)} scenarios")
    for i, s in enumerate(SCENARIOS[:5]):
        print(f"  [{i}] {s}")
