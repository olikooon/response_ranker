"""
Synthetic conversation dataset for response ranking.

Each example contains a multi-turn context and a pool of candidates:
one gold (correct) response plus distractors drawn from other conversations.

"""

import random
from dataclasses import dataclass, field
from typing import List, Tuple

random.seed(42)


@dataclass
class ConversationExample:
    context: str          # concatenated conversation history
    gold: str             # correct next utterance
    distractors: List[str] = field(default_factory=list)

    def candidates(self, shuffle: bool = True) -> Tuple[List[str], int]:
        """Return (candidate_list, gold_index). Gold is always included."""
        pool = [self.gold] + self.distractors
        if shuffle:
            random.shuffle(pool)
        gold_idx = pool.index(self.gold)
        return pool, gold_idx


# ---------------------------------------------------------------------------
# Raw conversation data: (context_turns, gold_response)
# Domains: IT support, customer service, health, education,
#          cooking, finance, travel, fitness, chitchat, mental health
# ---------------------------------------------------------------------------

_RAW: List[Tuple[List[str], str]] = [
    # ── IT Support ──────────────────────────────────────────────────────────
    (
        ["User: My laptop screen goes black randomly and won't wake up.",
         "Agent: Does it happen only on battery or also when plugged in?",
         "User: Both. It started after a Windows update last week."],
        "Agent: A recent Windows update could have changed your power or display driver settings. "
        "Go to Device Manager, expand Display Adapters, right-click your GPU, and select "
        "'Update driver'. Also check Power Options and disable 'Adaptive brightness'.",
    ),
    (
        ["User: My WiFi keeps disconnecting every 10 minutes.",
         "Agent: Is it only on your device or do others disconnect too?",
         "User: Only on my laptop. Phone and tablet stay connected fine."],
        "Agent: Since only your laptop disconnects, try disabling WiFi power management. "
        "In Device Manager, right-click your network adapter, go to Properties → "
        "Power Management, and uncheck 'Allow the computer to turn off this device to save power'.",
    ),
    (
        ["User: I accidentally deleted an important folder. Can I recover it?",
         "Agent: Did you empty the Recycle Bin after deleting it?",
         "User: Yes, I emptied it already."],
        "Agent: Don't write any new files to that drive to avoid overwriting the data. "
        "Use a recovery tool like Recuva or TestDisk. Both are free and can often "
        "retrieve files even after the Recycle Bin is emptied.",
    ),
    (
        ["User: My SSD is not detected in BIOS after I reseated it.",
         "Agent: Which slot did you put it back into?",
         "User: The same M.2 slot it was in before."],
        "Agent: Try clearing the CMOS by removing the motherboard battery for 30 seconds. "
        "Also check that the M.2 screw is firmly tightening the drive at a slight angle — "
        "a loose screw is a common cause of intermittent detection.",
    ),
    (
        ["User: Python crashes with 'MemoryError' when processing a large CSV.",
         "Agent: How large is the CSV and how much RAM does your machine have?",
         "User: It's 8 GB and I have 16 GB RAM."],
        "Agent: Loading an 8 GB CSV entirely into pandas can exceed available RAM due to overhead. "
        "Use `pd.read_csv(..., chunksize=100000)` to process the file in chunks, "
        "or switch to Polars or Dask which handle large files more efficiently.",
    ),

    # ── Customer Service ─────────────────────────────────────────────────────
    (
        ["Customer: I ordered a jacket 10 days ago and tracking still shows 'processing'.",
         "Agent: I'm sorry for the delay. May I have your order number?",
         "Customer: It's ORD-884521."],
        "Agent: Thank you. I can see your order was held due to a stock discrepancy. "
        "I've escalated it to our warehouse team and you'll receive a dispatch "
        "confirmation within 24 hours. We'll also apply a 10% discount to your next order.",
    ),
    (
        ["Customer: I was charged twice for the same subscription.",
         "Agent: I apologize for that. Can you confirm the email on the account?",
         "Customer: Yes, it's john.doe@email.com."],
        "Agent: I can confirm a duplicate charge on 14 April. I've initiated a full refund "
        "for the duplicate transaction — it should appear on your statement within 3–5 business days. "
        "You'll receive a confirmation email shortly.",
    ),
    (
        ["Customer: The product I received is completely different from what I ordered.",
         "Agent: I'm very sorry. Could you describe what you received?",
         "Customer: I ordered a blue mug but got a red vase."],
        "Agent: That is clearly a picking error on our end. I'll send a prepaid return label "
        "to your email immediately, and ship the correct blue mug via express delivery at no cost. "
        "You don't need to wait for us to receive the return before we dispatch the replacement.",
    ),
    (
        ["Customer: My promo code says it's invalid at checkout.",
         "Agent: Is the code case-sensitive when you enter it?",
         "Customer: I've tried both uppercase and lowercase."],
        "Agent: The code SUMMER20 expired on April 15th. As a courtesy I'll manually apply "
        "a 20% discount to your current cart. Please don't close the session and "
        "I'll update the order on my end right now.",
    ),
    (
        ["Customer: I want to cancel my gym membership but the app won't let me.",
         "Agent: Are you seeing a specific error message or just no cancel button?",
         "Customer: There's just no cancel button anywhere in settings."],
        "Agent: Cancellation must currently be done through our front desk or by calling "
        "our support line at 1-800-555-0199 during business hours. I know that's not ideal — "
        "we're adding in-app cancellation next month. I can also email you a cancellation form right now.",
    ),

    # ── Health & Medical ──────────────────────────────────────────────────────
    (
        ["Patient: I've had a headache every morning for the past week.",
         "Doctor: Is the pain worse on one side or both?",
         "Patient: Mostly on the right side, behind my eye."],
        "Doctor: A unilateral headache behind the eye, especially recurring in the morning, "
        "can indicate cluster headaches or tension-type headaches. I'd like to check your "
        "blood pressure today and ask about your sleep quality. Have you changed your screen time recently?",
    ),
    (
        ["Patient: I've been feeling very tired even after 8 hours of sleep.",
         "Doctor: How long has this been going on?",
         "Patient: About three months now. Also gained some weight."],
        "Doctor: Fatigue combined with unexplained weight gain over three months warrants "
        "a thyroid panel. Hypothyroidism fits both symptoms very well. "
        "I'll order a TSH, Free T4, and a CBC to rule out anaemia as well.",
    ),
    (
        ["Patient: My lower back has been aching since I started working from home.",
         "Doctor: Do you have any numbness or tingling down your legs?",
         "Patient: No, just the dull ache across my lower back."],
        "Doctor: This sounds like postural lower back pain — very common with prolonged desk work. "
        "I recommend raising your monitor to eye level, using a lumbar-support cushion, "
        "and doing 5 minutes of hip flexor stretches every hour. I can also refer you to physiotherapy.",
    ),
    (
        ["Patient: I have a rash on my arm that appeared two days ago. It's itchy.",
         "Doctor: Is the rash spreading or staying in the same spot?",
         "Patient: It's spreading slowly toward my elbow."],
        "Doctor: A spreading itchy rash could be contact dermatitis, ringworm, or in some cases "
        "an early sign of Lyme disease if you've been outdoors recently. "
        "I'd like to examine it in person — can you come in today or tomorrow? "
        "In the meantime, avoid scratching and apply a cool compress.",
    ),
    (
        ["Patient: Should I take antibiotics for a cold I've had for five days?",
         "Doctor: Are you experiencing green/yellow mucus, sore throat, or fever?",
         "Patient: Just a runny nose and congestion, no fever."],
        "Doctor: Colds are caused by viruses, so antibiotics won't help and can cause side effects. "
        "Stay hydrated, use saline nasal rinse, and rest. "
        "If symptoms worsen or a fever develops after day 7, come back in — "
        "that may signal a secondary bacterial infection.",
    ),

    # ── Education / Tutoring ─────────────────────────────────────────────────
    (
        ["Student: I don't understand when to use the definite article 'the' in English.",
         "Tutor: Can you give me a sentence where you're unsure?",
         "Student: 'I saw the dog' vs 'I saw a dog' — when do I use which?"],
        "Tutor: Great example. Use 'a dog' the first time you mention it — it's new information. "
        "Use 'the dog' afterwards because both speaker and listener now know which dog. "
        "Think of 'the' as pointing to something already established in shared context.",
    ),
    (
        ["Student: I keep mixing up correlation and causation in statistics.",
         "Tutor: That's a very common confusion. What's your current understanding?",
         "Student: I think correlation means two things happen together?"],
        "Tutor: Exactly — correlation means two variables move together, but one doesn't cause the other. "
        "Classic example: ice cream sales and drowning rates both rise in summer. "
        "The hidden cause is hot weather. Always ask: 'is there a third variable explaining both?' "
        "before claiming causation.",
    ),
    (
        ["Student: I don't know how to start my essay introduction.",
         "Tutor: What is your essay topic?",
         "Student: The impact of social media on teenagers' mental health."],
        "Tutor: Open with a hook — a striking statistic works well here, like 'teens who spend "
        "5+ hours daily on social media are 3× more likely to report depression'. "
        "Then narrow down to your thesis. Avoid starting with 'In today's world...' — "
        "it's overused and weakens your opening.",
    ),
    (
        ["Student: I'm really struggling with integration by parts.",
         "Tutor: Do you remember the formula: ∫u dv = uv - ∫v du?",
         "Student: I know the formula but I don't know how to pick u and v."],
        "Tutor: Use the LIATE rule to pick u: Logarithms, Inverse trig, Algebraic (polynomials), "
        "Trig, Exponential — choose u from whichever comes first in that list. "
        "For ∫x·eˣ dx, x is algebraic and eˣ is exponential, so u = x, dv = eˣ dx.",
    ),
    (
        ["Student: How do I stop procrastinating on revision?",
         "Tutor: Have you tried breaking study sessions into smaller blocks?",
         "Student: I start strong but then get distracted after 10 minutes."],
        "Tutor: 10 minutes is a sign your tasks feel too large and overwhelming. "
        "Try the Pomodoro technique: 25-minute focused sprint, 5-minute break. "
        "More importantly, write down the single next action before you start — "
        "'read page 45 to 60' beats 'study Chapter 3'.",
    ),

    # ── Cooking ──────────────────────────────────────────────────────────────
    (
        ["User: My pasta sauce always tastes too acidic.",
         "Chef: Are you using canned tomatoes or fresh ones?",
         "User: Canned. San Marzano brand."],
        "Chef: San Marzano are great but still acidic. Add a small pinch of baking soda "
        "(¼ tsp per 400 g can) right after the tomatoes go in — it neutralizes acidity without "
        "adding sweetness. Alternatively, simmer for 45+ minutes; long cooking mellows acidity naturally.",
    ),
    (
        ["User: My scrambled eggs turn out rubbery.",
         "Chef: What heat are you cooking them on?",
         "User: Medium-high, and I add milk."],
        "Chef: Medium-high is too hot — it makes proteins seize up fast. Switch to low heat, "
        "use a rubber spatula, and stir constantly. Pull them off the heat when they still look "
        "slightly underdone; carry-over heat finishes them. Skip the milk — it dilutes flavour and adds steam.",
    ),
    (
        ["User: How do I get a crispy crust on pan-seared chicken?",
         "Chef: Is your pan preheated before you add the chicken?",
         "User: I usually put the chicken in while the pan is heating."],
        "Chef: That's the issue. Preheat a stainless or cast-iron pan over medium-high for 2 minutes "
        "until a drop of water skips across. Pat the chicken completely dry, season, then lay it "
        "skin-side down without touching it for 5–6 minutes. Resist moving it — it releases naturally when crust forms.",
    ),
    (
        ["User: My bread doesn't rise properly. It stays flat.",
         "Chef: What yeast are you using and how old is it?",
         "User: Instant yeast, opened a few months ago."],
        "Chef: Old yeast is the likely culprit. Test it: dissolve 1 tsp in warm water (40°C) "
        "with a pinch of sugar — it should foam in 10 minutes. If it doesn't, the yeast is dead. "
        "Also ensure your water isn't too hot (above 50°C kills yeast).",
    ),
    (
        ["User: Can I substitute Greek yogurt for sour cream in baking?",
         "Chef: Are you using it in a dip or baking something?",
         "User: In a chocolate cake recipe."],
        "Chef: Yes, full-fat Greek yogurt is a 1:1 substitute in baking. "
        "The fat content keeps the cake moist and the acidity reacts with baking soda "
        "just like sour cream. Low-fat yogurt works too but may produce a slightly denser crumb.",
    ),

    # ── Finance ──────────────────────────────────────────────────────────────
    (
        ["Client: I'm 28 and just started investing. Where should I begin?",
         "Advisor: Do you have an emergency fund of 3–6 months' expenses first?",
         "Client: Yes, I have 4 months saved up."],
        "Advisor: Great foundation. At 28 with a long horizon, prioritize your employer's 401(k) "
        "up to the full match — that's a guaranteed 50–100% return. Then max your Roth IRA ($7,000/yr). "
        "For both, use low-cost total-market index funds; Vanguard VTSAX or Fidelity FZROX are solid choices.",
    ),
    (
        ["Client: My credit score dropped 40 points after paying off a loan.",
         "Advisor: Did you close the account after paying it off?",
         "Client: Yes, I called and closed it right away."],
        "Advisor: Closing the account reduced your 'length of credit history' and "
        "available credit, both of which affect your score. For installment loans, "
        "paying them off is still correct — the score dip is temporary (3–6 months) "
        "and your overall credit profile will improve.",
    ),
    (
        ["Client: Should I pay off my student loans early or invest the extra money?",
         "Advisor: What's your interest rate on the loans?",
         "Client: 5.5% fixed."],
        "Advisor: At 5.5%, the decision is close. If your investments can realistically earn 7–8% "
        "(historical S&P 500 average), investing wins mathematically in the long run. "
        "A common middle ground: invest up to any employer 401(k) match, then split extra cash "
        "50/50 between the loan and a Roth IRA.",
    ),
    (
        ["Client: I keep overspending and running out of money before month end.",
         "Advisor: Do you track your spending in any way currently?",
         "Client: Not really, I just watch my bank balance."],
        "Advisor: Watching the balance isn't enough because it doesn't show categories. "
        "Try the 50/30/20 rule: 50% needs (rent, food), 30% wants (dining, entertainment), "
        "20% savings/debt. Tools like YNAB or Monarch Money auto-categorize transactions — "
        "most people are shocked where their money actually goes.",
    ),
    (
        ["Client: Is it safe to keep all my savings in one bank?",
         "Advisor: How much total savings are you holding?",
         "Client: Around $180,000."],
        "Advisor: FDIC insurance covers up to $250,000 per depositor per bank, so $180,000 "
        "in one FDIC-insured bank is fully protected. However, spreading across two banks "
        "gives you redundancy if one has a service outage or account freeze issue — "
        "a practical precaution regardless of insurance limits.",
    ),

    # ── Travel ───────────────────────────────────────────────────────────────
    (
        ["Traveler: I'm visiting Japan for the first time in April. Any tips?",
         "Guide: How long will you be there and which cities?",
         "Traveler: Two weeks — Tokyo, Kyoto, and Osaka."],
        "Guide: Perfect itinerary. April is cherry blossom season — book accommodation now, "
        "prices triple and rooms sell out. Get an IC Card (Suica or ICOCA) at the airport "
        "for seamless transit. In Kyoto, rent a bicycle to reach Fushimi Inari at dawn "
        "before crowds arrive. Reserve a kaiseki dinner in advance; they fill up fast.",
    ),
    (
        ["Traveler: What's the safest way to carry money abroad?",
         "Guide: Are you going to a region with good ATM coverage?",
         "Traveler: Visiting rural Southeast Asia, so ATMs may be sparse."],
        "Guide: Carry a mix: use a Charles Schwab debit card (reimburses ATM fees worldwide) "
        "for towns with ATMs, and keep 3–4 days of cash in local currency as backup. "
        "Split cash between a money belt and your bag — never keep it all together. "
        "Avoid airport exchange kiosks; they charge 10–15% above market rate.",
    ),
    (
        ["Traveler: I got food poisoning abroad. What should I do?",
         "Guide: Are you able to keep fluids down?",
         "Traveler: Barely. Everything I drink comes back up."],
        "Guide: If you can't keep fluids down for 12+ hours, see a doctor — dehydration risk "
        "is serious in hot climates. Most hotels have a doctor on call. "
        "Oral rehydration salts (ORS) work faster than plain water; pharmacies carry them. "
        "Avoid anti-diarrheal medication in the first 24 hours unless a doctor prescribes it.",
    ),
    (
        ["Traveler: My flight got cancelled and the airline is offering a full refund or a voucher.",
         "Guide: Is the voucher worth significantly more than the refund?",
         "Traveler: Voucher is $50 more but expires in 6 months."],
        "Guide: Take the refund unless you're certain you'll fly that airline within 6 months. "
        "Voucher blackout dates and restrictions often make the extra $50 worthless. "
        "Under EC 261 (if flying from or to the EU), you may also be entitled to additional "
        "compensation depending on notice time and distance.",
    ),
    (
        ["Traveler: How do I avoid jet lag flying from New York to Tokyo?",
         "Guide: What time does your flight depart?",
         "Traveler: Evening flight, landing Tokyo morning local time."],
        "Guide: On the plane, set your watch to Tokyo time immediately and sleep "
        "during Tokyo's night hours — use an eye mask and earplugs. "
        "After landing, stay awake until 9–10 PM local time regardless of tiredness; "
        "get sunlight exposure in the afternoon, which resets your circadian clock fastest.",
    ),

    # ── Fitness ──────────────────────────────────────────────────────────────
    (
        ["User: I've been running 5k three times a week but my time isn't improving.",
         "Coach: How long have you been at this routine?",
         "User: About four months."],
        "Coach: Four months without progression means your body has adapted — you need new stimulus. "
        "Add one interval session per week: 8×400 m at hard effort with 90-second recoveries. "
        "Keep your other two runs easy (conversational pace). Speed work is what actually improves 5k time.",
    ),
    (
        ["User: I want to build muscle but I don't want to get too bulky.",
         "Coach: Are you currently doing any resistance training?",
         "User: No, just cardio. I'm afraid of lifting weights."],
        "Coach: 'Getting bulky' requires years of consistent heavy lifting plus a significant calorie surplus — "
        "it won't happen by accident. Resistance training with moderate weight and 8–12 reps "
        "builds lean, defined muscle. Start with compound movements: squats, rows, push-ups. "
        "It will make your cardio more effective too.",
    ),
    (
        ["User: My knees hurt when I do squats.",
         "Coach: Does the pain start right away or after a few reps?",
         "User: It starts immediately when I go below parallel."],
        "Coach: Pain below parallel usually means one of three things: tight hip flexors limiting depth, "
        "heels rising (put 10 lb plates under them temporarily to diagnose), "
        "or knees caving inward. Film yourself from the side and front. "
        "In the meantime, try box squats to a depth just above parallel while you address mobility.",
    ),
    (
        ["User: I'm losing weight but I also seem to be losing muscle.",
         "Coach: How large is your daily calorie deficit?",
         "User: I'm eating about 1,000 calories below maintenance."],
        "Coach: A 1,000-calorie deficit is too aggressive and is causing muscle breakdown. "
        "Cut it to 300–500 calories below maintenance. More importantly, increase protein to "
        "at least 2.2 g per kg of bodyweight — that's the key lever for preserving muscle during a cut. "
        "Keep lifting heavy; don't switch to high-rep 'toning' work.",
    ),
    (
        ["User: I get really sore for 3 days after leg day and can't train.",
         "Coach: Are you new to training or returning after a break?",
         "User: I've been lifting consistently for a year."],
        "Coach: After a year, 3-day soreness suggests you're doing too much volume in one session. "
        "Split leg work across two days (quads/glutes one day, hamstrings/calves another). "
        "Also try 10 minutes of light cycling immediately post-workout — blood flow "
        "clears metabolic waste and cuts DOMS severity significantly.",
    ),

    # ── Chitchat ─────────────────────────────────────────────────────────────
    (
        ["Alice: I've been reading a lot lately. Any book recommendations?",
         "Bob: What genres do you usually enjoy?",
         "Alice: I love thrillers and anything that makes me think."],
        "Bob: Then you'd love 'Dark Matter' by Blake Crouch — it's a fast-paced sci-fi thriller "
        "with genuine philosophical depth about identity and choice. "
        "Also try 'The Thursday Murder Club' by Richard Osman if you want something "
        "lighter but still sharp.",
    ),
    (
        ["Alice: I've been really stressed out with work lately.",
         "Bob: That sounds tough. Is it the workload or something else?",
         "Alice: It's mostly my manager — he micromanages everything."],
        "Bob: Micromanagement is exhausting because it signals a lack of trust. "
        "It sometimes helps to proactively over-communicate your progress before he asks — "
        "it can reduce his anxiety and give you more autonomy. "
        "Have you tried a brief weekly update email to him outlining what you're working on?",
    ),
    (
        ["Alice: I'm thinking of learning a new language. Which one should I pick?",
         "Bob: Do you have a specific reason — travel, career, or just personal interest?",
         "Alice: Mainly travel. I love going to Europe."],
        "Bob: Spanish is the most practical for Europe and worldwide — widely spoken in Spain "
        "and gives you access to Latin America too. "
        "Italian is easier for English speakers and incredibly rewarding in Italy. "
        "If France is your top destination, French pairs well with travel to North Africa as a bonus.",
    ),
    (
        ["Alice: I've been watching too much TV and feeling guilty about it.",
         "Bob: How many hours a day roughly?",
         "Alice: Probably 4–5 hours in the evenings."],
        "Bob: That's a significant chunk. Try the 'replacement' trick instead of cutting cold turkey — "
        "swap one hour of passive watching for something with mild engagement: an audiobook during a walk, "
        "or a podcast while doing chores. You end up with the same downtime but feel less zombie-like.",
    ),
    (
        ["Alice: I just moved to a new city and don't know anyone.",
         "Bob: How long have you been there?",
         "Alice: About two months. I work remotely so I don't meet people at work."],
        "Bob: Remote work makes this genuinely harder — you have to be intentional. "
        "Meetup.com groups around hobbies (climbing, board games, language exchange) are consistently "
        "the fastest way to meet people with shared interests. "
        "Also look for local sports leagues; recreational volleyball or ultimate frisbee "
        "teams are very welcoming to newcomers.",
    ),

    # ── Mental Health ─────────────────────────────────────────────────────────
    (
        ["Client: I've been feeling really anxious about everything lately.",
         "Therapist: When you say 'everything', can you give me an example from this week?",
         "Client: Even small things like sending an email at work make me spiral."],
        "Therapist: What you're describing — catastrophizing ordinary tasks — is very common in "
        "generalized anxiety. One technique that helps is asking yourself: 'What's the worst realistic outcome, "
        "and could I handle it?' Usually the answer to the second question is yes, which interrupts the spiral. "
        "Would you like to practice this with the email example?",
    ),
    (
        ["Client: I feel like I'm always tired and nothing is fun anymore.",
         "Therapist: How long have you been feeling this way?",
         "Client: A few months. I used to love hiking but now I can't be bothered."],
        "Therapist: Losing interest in activities you used to enjoy — called anhedonia — "
        "combined with persistent fatigue is something I'd like to explore further. "
        "These can be signs of depression. How is your sleep, and have you noticed any changes in appetite? "
        "I want to make sure we get the full picture before we talk about next steps.",
    ),
    (
        ["Client: I have a big presentation at work and I can't sleep because of it.",
         "Therapist: Is this a recurring pattern before important events?",
         "Client: Yes, I've always been like this before big things."],
        "Therapist: Pre-event insomnia from performance anxiety is very treatable. "
        "Try a 'worry dump' 2 hours before bed: write every concern down and then physically close the notebook. "
        "This externalizes the thoughts so your brain stops rehearsing them. "
        "Also, a 4-7-8 breathing exercise (inhale 4, hold 7, exhale 8) activates your parasympathetic system.",
    ),
    (
        ["Client: I feel guilty spending any time on myself.",
         "Therapist: Where do you think that guilt comes from?",
         "Client: My parents always said that resting was lazy."],
        "Therapist: That's a deeply ingrained belief — 'rest equals laziness' — and it sounds like "
        "it's been with you since childhood. It's worth examining: do you apply that same standard to people you love? "
        "Self-care isn't a reward for productivity; it's maintenance that makes everything else sustainable. "
        "What would feel like a safe, small act of self-care you could try this week?",
    ),
    (
        ["Client: I snap at my partner over small things and then feel terrible.",
         "Therapist: What's usually happening in the moments just before you snap?",
         "Client: I think I'm already stressed from work and they just add one more thing."],
        "Therapist: That's a classic stress-displacement pattern — unprocessed work stress "
        "spills over at home because home feels safer to express emotions. "
        "Creating a 10-minute transition ritual when you finish work (walk, tea, music) "
        "can act as a pressure valve before you re-engage with your partner. "
        "Would it help to tell your partner what's going on so they understand the context?",
    ),
]


def build_dataset(num_distractors: int = 9) -> List[ConversationExample]:
    """
    Build ConversationExample objects. Distractors are gold responses
    from other conversations a standard negative-sampling strategy
    used in benchmarks like Ubuntu Corpus.
    """
    examples = []
    all_gold = [gold for _, gold in _RAW]

    for i, (turns, gold) in enumerate(_RAW):
        context = " | ".join(turns)

        # Pick distractors from all other gold responses
        distractor_pool = [g for j, g in enumerate(all_gold) if j != i]
        num_d = min(num_distractors, len(distractor_pool))
        distractors = random.sample(distractor_pool, num_d)

        examples.append(ConversationExample(
            context=context,
            gold=gold,
            distractors=distractors,
        ))

    return examples


def get_pool_size(examples: List[ConversationExample]) -> int:
    return 1 + len(examples[0].distractors)
